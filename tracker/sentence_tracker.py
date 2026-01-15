"""
sentence_matcher.py
문장 유사도 기반 키워드 매칭

사용법:
    matcher = SentenceMatcher(keywords, embedder)
    matches = matcher.find_matches("분석 결과 t-SNE로 시각화했습니다", threshold=0.7)
    
    for kw, score in matches:
        print(f"{kw}: {score:.3f}")
        
특징:
- 키워드 임베딩 사전 계산 (한 번만)
- 문장 입력 시 모든 키워드와 유사도 계산
- 임계값 이상만 반환
- 점수 순 정렬
"""
from __future__ import annotations

import math
from typing import List, Tuple, Optional, Dict, Any
from dataclasses import dataclass


def cosine_similarity(vec1: List[float], vec2: List[float]) -> float:
    """코사인 유사도 계산"""
    if len(vec1) != len(vec2) or not vec1:
        return 0.0
    
    dot = sum(x * y for x, y in zip(vec1, vec2))
    norm1 = math.sqrt(sum(x * x for x in vec1))
    norm2 = math.sqrt(sum(x * x for x in vec2))
    
    if norm1 == 0 or norm2 == 0:
        return 0.0
    
    return dot / (norm1 * norm2)


@dataclass
class KeywordMatch:
    """키워드 매칭 결과"""
    keyword: str
    score: float
    slide_no: Optional[int] = None
    
    def __repr__(self) -> str:
        slide_info = f" (Slide {self.slide_no})" if self.slide_no else ""
        return f"KeywordMatch('{self.keyword}'{slide_info}, score={self.score:.3f})"


class SentenceMatcher:
    """
    문장 유사도 기반 키워드 매칭
    
    직접 유사도 비교 방식 (키워드 140개 정도에 최적)
    - FAISS보다 간단하고 정확
    - 키워드가 수천 개 이하면 충분히 빠름
    """
    
    def __init__(self, embedder, keywords: Optional[List[str]] = None, 
                 slide_keywords: Optional[Dict[int, List[str]]] = None):
        """
        Args:
            embedder: UpstageEmbedder 또는 TransformerEmbedder
            keywords: 전체 키워드 리스트
            slide_keywords: 슬라이드별 키워드 딕셔너리 {slide_no: [keywords]}
        """
        self.embedder = embedder
        self.keywords: List[str] = keywords or []
        self.slide_keywords = slide_keywords or {}
        
        # 키워드 임베딩 캐시 {keyword: embedding}
        self._keyword_embeddings: Dict[str, List[float]] = {}
        
        # 슬라이드별 키워드도 추가
        if slide_keywords:
            for kws in slide_keywords.values():
                self.keywords.extend(kws)
            # 중복 제거
            self.keywords = list(set(self.keywords))
        
        # 키워드 임베딩 사전 계산
        if self.keywords:
            self._precompute_embeddings()
    
    def _precompute_embeddings(self) -> None:
        """키워드 임베딩 사전 계산 (한 번만 실행)"""
        print(f"   키워드 임베딩 계산 중... ({len(self.keywords)}개)")
        
        for i, kw in enumerate(self.keywords, 1):
            if kw not in self._keyword_embeddings:
                self._keyword_embeddings[kw] = self.embedder.embed(kw)
                
            if i % 20 == 0:
                print(f"      진행: {i}/{len(self.keywords)}")
        
        print(f"   키워드 임베딩 완료")
    
    def find_matches(
        self, 
        sentence: str, 
        threshold: float = 0.5,
        top_k: Optional[int] = None,
        slide_no: Optional[int] = None
    ) -> List[KeywordMatch]:
        """
        문장과 유사한 키워드 찾기
        
        Args:
            sentence: 입력 문장
            threshold: 최소 유사도 (0~1)
            top_k: 상위 k개만 반환 (None이면 전체)
            slide_no: 특정 슬라이드 키워드만 검색
            
        Returns:
            KeywordMatch 리스트 (점수 높은 순 정렬)
        """
        # 문장 임베딩
        sentence_emb = self.embedder.embed(sentence)
        
        # 검색 대상 키워드 선택
        if slide_no and slide_no in self.slide_keywords:
            target_keywords = self.slide_keywords[slide_no]
        else:
            target_keywords = self.keywords
        
        # 모든 키워드와 유사도 계산
        matches = []
        for kw in target_keywords:
            if kw not in self._keyword_embeddings:
                # 캐시에 없으면 계산
                self._keyword_embeddings[kw] = self.embedder.embed(kw)
            
            kw_emb = self._keyword_embeddings[kw]
            score = cosine_similarity(sentence_emb, kw_emb)
            
            if score >= threshold:
                matches.append(KeywordMatch(
                    keyword=kw,
                    score=score,
                    slide_no=slide_no
                ))
        
        # 점수 순 정렬
        matches.sort(key=lambda x: x.score, reverse=True)
        
        # top_k 제한
        if top_k:
            matches = matches[:top_k]
        
        return matches
    
    def batch_match(
        self,
        sentences: List[str],
        threshold: float = 0.7,
        top_k: Optional[int] = None
    ) -> List[List[KeywordMatch]]:
        """
        여러 문장에 대해 배치 매칭
        
        Args:
            sentences: 문장 리스트
            threshold: 최소 유사도
            top_k: 문장당 상위 k개
            
        Returns:
            각 문장별 매칭 결과 리스트
        """
        results = []
        for sentence in sentences:
            matches = self.find_matches(sentence, threshold, top_k)
            results.append(matches)
        return results
    
    def get_coverage_for_slide(
        self,
        slide_no: int,
        script_text: str,
        threshold: float = 0.7
    ) -> Dict[str, Any]:
        """
        특정 슬라이드의 키워드 커버리지 계산
        
        Args:
            slide_no: 슬라이드 번호
            script_text: 대본 텍스트
            threshold: 최소 유사도
            
        Returns:
            {
                'total': 전체 키워드 수,
                'covered': 커버된 키워드 수,
                'coverage': 커버율 (0~1),
                'matched_keywords': [(keyword, score), ...],
                'uncovered_keywords': [keyword, ...]
            }
        """
        if slide_no not in self.slide_keywords:
            return {
                'total': 0,
                'covered': 0,
                'coverage': 0.0,
                'matched_keywords': [],
                'uncovered_keywords': []
            }
        
        target_keywords = self.slide_keywords[slide_no]
        matches = self.find_matches(script_text, threshold, slide_no=slide_no)
        
        matched_kw_set = {m.keyword for m in matches}
        uncovered = [kw for kw in target_keywords if kw not in matched_kw_set]
        
        return {
            'total': len(target_keywords),
            'covered': len(matched_kw_set),
            'coverage': len(matched_kw_set) / len(target_keywords) if target_keywords else 0.0,
            'matched_keywords': [(m.keyword, m.score) for m in matches],
            'uncovered_keywords': uncovered
        }
    
    def interactive_mode(self, threshold: float = 0.7, top_k: int = 5):
        """
        대화형 모드: 문장 입력 > 매칭 > 결과 출력
        
        종료: 'quit', 'exit', 'q' 입력
        """
        print("=" * 80)
        print("문장 유사도 매칭 (Interactive Mode)")
        print("=" * 80)
        print(f"임계값: {threshold:.2f} | 상위 {top_k}개 표시")
        print(f"총 키워드: {len(self.keywords)}개")
        print("종료: quit, exit, q")
        print("=" * 80)
        print()
        
        while True:
            try:
                sentence = input("문장 입력> ").strip()
                
                if sentence.lower() in ['quit', 'exit', 'q', '']:
                    print("종료합니다.")
                    break
                
                matches = self.find_matches(sentence, threshold, top_k)
                
                if matches:
                    print(f"\n{len(matches)}개 매칭됨:")
                    for i, match in enumerate(matches, 1):
                        print(f"  {i}. {match.keyword}: {match.score:.3f}")
                else:
                    print(f"\n매칭 없음 (임계값 {threshold} 이상 없음)")
                
                print()
                
            except KeyboardInterrupt:
                print("\n\n종료합니다.")
                break
            except Exception as e:
                print(f"오류: {e}")
                continue


# FAISS 대안 (참고용 - 나중에 키워드가 수천 개로 늘어나면 사용)
"""
FAISS 사용 예시 (키워드 1만 개 이상일 때):

import faiss
import numpy as np

class FAISSMatcher:
    def __init__(self, embedder, keywords):
        self.embedder = embedder
        self.keywords = keywords
        
        # 임베딩 계산
        embeddings = [embedder.embed(kw) for kw in keywords]
        embeddings_np = np.array(embeddings, dtype='float32')
        
        # FAISS 인덱스 생성
        dim = len(embeddings[0])
        self.index = faiss.IndexFlatIP(dim)  # Inner Product (cosine sim)
        
        # 정규화 (코사인 유사도용)
        faiss.normalize_L2(embeddings_np)
        self.index.add(embeddings_np)
    
    def find_matches(self, sentence, threshold=0.7, top_k=10):
        sentence_emb = self.embedder.embed(sentence)
        sentence_emb_np = np.array([sentence_emb], dtype='float32')
        faiss.normalize_L2(sentence_emb_np)
        
        # 검색
        scores, indices = self.index.search(sentence_emb_np, top_k)
        
        matches = []
        for score, idx in zip(scores[0], indices[0]):
            if score >= threshold:
                matches.append((self.keywords[idx], float(score)))
        
        return matches
"""
