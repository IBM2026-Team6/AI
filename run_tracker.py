"""
키워드 추적 분석 시스템

사용법:
  python run_tracker.py -m hybrid                    # 하이브리드 모드, 정규화 ON (기본값)
  python run_tracker.py -m token --normalize n       # 토큰 모드, 정규화 OFF
  python run_tracker.py -m sentence --api y          # 문장 모드, Upstage API
"""
from __future__ import annotations

import argparse
import re
from pathlib import Path
from typing import List, Dict

from config import Config
from tracker.keyword_tracker import (
    SlideKeywordTracker,
    TrackerConfig,
    UpstageEmbedder,
    TransformerEmbedder,
    EmbedderConfig,
    parse_scripts_by_slide,
    extract_keywords_from_script,
    SlideAnalysis,
    save_coverage_report,
)
from tracker.sentence_tracker import SentenceMatcher


def normalize_keyword(keyword: str, mode: str = "space") -> str:
    """
    키워드 정규화
    
    Args:
        keyword: 원본 키워드
        mode: "remove" (제거), "space" (공백으로 치환)
    
    Returns:
        정규화된 키워드
    """
    if mode == "remove":
        # 하이픈, 언더스코어 제거: "운영-최적화" → "운영최적화"
        return re.sub(r'[-_]', '', keyword)
    elif mode == "space":
        # 하이픈, 언더스코어를 공백으로: "운영-최적화" → "운영 최적화"
        return re.sub(r'[-_]', ' ', keyword)
    else:
        return keyword


def parse_keywords_from_file(
    keywords_file: str, 
    normalize: bool = False,
    normalize_mode: str = "space"
) -> Dict[int, List[str]]:
    """
    키워드 파일 파싱 (정규화 옵션 포함)
    
    Args:
        keywords_file: 키워드 파일 경로
        normalize: 정규화 여부
        normalize_mode: "remove" 또는 "space"
    """
    slide_keywords: Dict[int, List[str]] = {}

    with open(keywords_file, "r", encoding="utf-8") as f:
        content = f.read()

    slide_blocks = re.findall(
        r"## Slide (\d+)\s*\n키워드 \(\d+개\):\s*\n((?:\s+\d+\..+\n)*)",
        content,
    )

    for slide_str, keywords_str in slide_blocks:
        slide_no = int(slide_str)
        keywords = re.findall(r"\d+\.\s+(.+?)$", keywords_str, re.MULTILINE)
        
        if normalize:
            keywords = [normalize_keyword(kw, normalize_mode) for kw in keywords]
        
        slide_keywords[slide_no] = keywords

    return slide_keywords


def main_analysis(
    use_api: bool = False, 
    method: str = "token",
    normalize: bool = False,
    normalize_mode: str = "remove"
):
    """
    키워드 추적 (정규화 옵션 포함)
    
    Args:
        use_api: Upstage API 사용 여부
        method: token, sentence, hybrid
        normalize: 키워드 정규화 여부
        normalize_mode: remove 또는 space
    """
    cfg = Config()
    keywords_file = Path(cfg.out_dir) / "paper_keywords.txt"
    scripts_file = Path(cfg.out_dir) / "paper_scripts.md"
    
    if not keywords_file.exists():
        print(f"키워드 파일 없음: {keywords_file}")
        return
    
    if not scripts_file.exists():
        print(f"스크립트 파일 없음: {scripts_file}")
        return
    
    # 1. 키워드 파싱 (정규화 옵션 적용)
    print("키워드 파일 분석 중...")
    slide_keywords = parse_keywords_from_file(
        str(keywords_file),
        normalize=normalize,
        normalize_mode=normalize_mode
    )
    print(f"   {len(slide_keywords)}개 슬라이드 키워드 로드됨")
    
    if normalize:
        mode_desc = "제거" if normalize_mode == "remove" else "공백 치환"
        print(f"   키워드 정규화: ON (특수문자 {mode_desc})")
        # 샘플 출력
        if 2 in slide_keywords and len(slide_keywords[2]) > 0:
            print(f"   예시: {slide_keywords[2][0]}")
    
    # 2. 스크립트 파싱
    print("스크립트 파일 분석 중...")
    slide_scripts = parse_scripts_by_slide(str(scripts_file))
    print(f"   {len(slide_scripts)}개 슬라이드 대본 로드됨")
    
    # 3. Tracker 초기화
    print("Tracker 초기화 중...")
    embedder_cfg = EmbedderConfig.from_config(cfg)
    
    if use_api:
        print("   Using Upstage API embedder")
        embedder = UpstageEmbedder(embedder_cfg)
    else:
        print("   Using Transformer embedder (local, fast)")
        embedder = TransformerEmbedder(embedder_cfg)
    
    tracker_cfg = TrackerConfig.from_config(cfg)
    
    if method in ["sentence", "hybrid"]:
        print("   문장 유사도 매칭 초기화 중...")
        sentence_matcher = SentenceMatcher(
            embedder=embedder,
            slide_keywords=slide_keywords
        )
        print("   SentenceMatcher 준비 완료")
    
    if method in ["token", "hybrid"]:
        tracker = SlideKeywordTracker(
            slide_keywords=slide_keywords,
            slide_aliases={},
            cfg=tracker_cfg,
            embedder=embedder,
        )
        print("   TokenTracker 준비 완료")
    
    print()
    
    # 4. 슬라이드별 분석
    results: List[SlideAnalysis] = []
    total_accuracy = 0.0
    
    for slide_no in sorted(slide_keywords.keys()):
        if slide_no not in slide_scripts:
            continue
        
        script_text = slide_scripts[slide_no]
        extracted = extract_keywords_from_script(script_text)
        input_sentences = []
        
        if method == "sentence":
            sentences = re.split(r'[.!?,]\s+', script_text)
            sentences = [s.strip() for s in sentences if s.strip() and len(s.strip()) > 5]
            input_sentences = sentences
            
            all_matched = set()
            for sent in sentences:
                matches = sentence_matcher.find_matches(
                    sentence=sent,
                    threshold=tracker_cfg.semantic_threshold,
                    slide_no=slide_no
                )
                for m in matches:
                    all_matched.add(m.keyword)
            
            covered = list(all_matched)
            uncovered = [kw for kw in slide_keywords[slide_no] if kw not in all_matched]
            
        elif method == "hybrid":
            token_status = tracker.update_with_stt(slide_no, script_text)
            token_covered = {x["keyword"] for x in token_status if x["covered"]}
            
            sentences = re.split(r'[.!?,]\s+', script_text)
            sentences = [s.strip() for s in sentences if s.strip() and len(s.strip()) > 5]
            
            sentence_covered = set()
            for sent in sentences:
                matches = sentence_matcher.find_matches(
                    sentence=sent,
                    threshold=tracker_cfg.semantic_threshold,
                    slide_no=slide_no
                )
                for m in matches:
                    sentence_covered.add(m.keyword)
            
            all_covered = token_covered | sentence_covered
            print(f"\nSlide {slide_no}: 토큰={len(token_covered)}개, 문장={len(sentence_covered)}개, 합계={len(all_covered)}개")
            
            covered = list(all_covered)
            uncovered = [kw for kw in slide_keywords[slide_no] if kw not in all_covered]
            
        else:
            token_status = tracker.update_with_stt(slide_no, script_text)
            covered = [x["keyword"] for x in token_status if x["covered"]]
            uncovered = [x["keyword"] for x in token_status if not x["covered"]]
        
        result = SlideAnalysis(
            slide_no=slide_no,
            total_keywords=len(slide_keywords[slide_no]),
            extracted_keywords=input_sentences if method == "sentence" else extracted,
            covered_keywords=covered,
            uncovered_keywords=uncovered,
        )
        results.append(result)
        total_accuracy += result.coverage_accuracy
    
    # 5. 결과 출력
    print("\n" + "=" * 80)
    print("슬라이드별 키워드 커버 분석 결과")
    print("=" * 80 + "\n")
    
    for result in results:
        bar_length = 30
        filled = int(bar_length * result.coverage_accuracy)
        bar = "█" * filled + "░" * (bar_length - filled)
        
        print(f"\nSlide {result.slide_no}")

        if result.extracted_keywords and method != "sentence":
            print(f"   대본 키워드: {', '.join(result.extracted_keywords)}")

        print(f"   정확도: {result.coverage_accuracy:.0%} [{bar}]")
        print(f"   커버됨: {len(result.covered_keywords)}/{result.total_keywords}")

        if result.covered_keywords:
            print(f"   포함된 키워드:")
            for kw in result.covered_keywords:
                print(f"      - {kw}")
        
        if result.uncovered_keywords:
            print(f"   누락된 키워드:")
            for kw in result.uncovered_keywords:
                print(f"      - {kw}")
    
    # 6. 전체 요약
    print("\n" + "=" * 80)
    print("전체 분석 요약")
    print("=" * 80)
    
    total_slides = len(results)
    avg_accuracy = total_accuracy / total_slides if total_slides > 0 else 0.0
    total_keywords = sum(r.total_keywords for r in results)
    total_covered = sum(len(r.covered_keywords) for r in results)
    
    print(f"  총 슬라이드 수: {total_slides}개")
    print(f"  총 키워드 수: {total_keywords}개")
    print(f"  커버된 키워드: {total_covered}개 ({total_covered/total_keywords*100:.1f}%)")
    print(f"  평균 커버 정확도: {avg_accuracy:.1%}")
    
    if normalize:
        mode_desc = "제거" if normalize_mode == "remove" else "공백치환"
        print(f"  정규화 모드: {mode_desc}")
    
    # 7. 결과 파일 저장
    output_file = Path(cfg.out_dir) / "paper_coverage_analysis.txt"
    save_coverage_report(results, avg_accuracy, total_keywords, total_covered, output_file)
    print(f"\n상세 결과 저장: {output_file}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="키워드 커버 추적 (정규화 옵션 포함)")
    parser.add_argument(
        "--api",
        type=str,
        choices=["y", "n"],
        default="n",
        help="Upstage API 사용 여부 (y: Upstage, n: Transformer)"
    )
    parser.add_argument(
        "-m", "--method",
        type=str,
        choices=["token", "sentence", "hybrid"],
        default="hybrid",
        help="매칭 방식: token, sentence, hybrid"
    )
    parser.add_argument(
        "--normalize",
        type=str,
        choices=["y", "n"],
        default="y",
        help="키워드 정규화 여부 (y: 특수문자 제거/치환, n: 원본 사용)"
    )
    parser.add_argument(
        "--normalize-mode",
        type=str,
        choices=["remove", "space"],
        default="space",
        help="정규화 모드: remove (제거), space (공백 치환)"
    )
    
    args = parser.parse_args()
    
    use_api = (args.api == "y")
    normalize = (args.normalize == "y")
    
    main_analysis(
        use_api=use_api,
        method=args.method,
        normalize=normalize,
        normalize_mode=args.normalize_mode
    )
