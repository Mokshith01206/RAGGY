#hello4: Raggy Reflective Agent
#hello4: Self-evaluating RAG agent verifying retrieval confidence and reformulating weak queries

from typing import List, Dict, Any, Optional
from dataclasses import dataclass, field
from raggy.retrievers.hybrid import RaggyHybridRetriever, RaggyScoredNode


@dataclass
class RaggyResponse:
    #hello4: Standard response structure returned by Raggy engines
    response: str
    source_nodes: List[RaggyScoredNode] = field(default_factory=list)
    confidence_score: float = 0.0
    reflection_count: int = 0
    metadata: Dict[str, Any] = field(default_factory=dict)

    def __str__(self) -> str:
        return self.response


class RaggyReflectiveAgent:
    #hello4: Agent that scores retrieval confidence and re-retrieves if below threshold

    def __init__(
        self,
        retriever: RaggyHybridRetriever,
        confidence_threshold: float = 0.5,
        max_iterations: int = 2,
        llm: Optional[Any] = None,
    ):
        #hello4: Initialize agent parameters
        self.retriever = retriever
        self.confidence_threshold = confidence_threshold
        self.max_iterations = max_iterations
        self.llm = llm

    def _assess_confidence(self, query: str, nodes: List[RaggyScoredNode]) -> float:
        #hello4: Compute blended confidence from score and lexical alignment
        if not nodes:
            return 0.0

        top_score = nodes[0].score if nodes else 0.0
        query_words = set(query.lower().split())
        matched_words = 0

        combined_text = " ".join([n.text.lower() for n in nodes[:3]])
        for qw in query_words:
            if qw in combined_text:
                matched_words += 1

        overlap_ratio = matched_words / max(len(query_words), 1)
        #hello4: Normalize blended confidence
        confidence = min(1.0, (overlap_ratio * 0.7) + (min(top_score * 50.0, 1.0) * 0.3))
        return confidence

    def _reformulate_query(self, query: str, iteration: int) -> str:
        #hello4: Expand query terms for secondary retrieval passes
        import re
        clean = re.sub(r"[^\w\s]", "", query)
        terms = [t for t in clean.split() if len(t) > 3]
        if iteration == 1 and terms:
            return f"{query} {' '.join(terms)}"
        return f"detailed overview of {query}"

    def query(self, query_str: str) -> RaggyResponse:
        #hello4: Execute multi-hop reflective retrieval
        current_query = query_str
        all_nodes: List[RaggyScoredNode] = []
        confidence = 0.0
        iterations = 0

        for it in range(self.max_iterations):
            iterations = it + 1
            #hello4: Retrieve candidate nodes
            nodes = self.retriever.retrieve(current_query)
            if nodes:
                all_nodes = nodes
                confidence = self._assess_confidence(current_query, nodes)
                #hello4: Check if confidence threshold is satisfied
                if confidence >= self.confidence_threshold:
                    break

            #hello4: Reformulate and re-retrieve
            current_query = self._reformulate_query(query_str, iterations)

        #hello4: Synthesize response
        if all_nodes:
            context_str = "\n\n---\n\n".join([f"[{n.metadata.get('breadcrumb', 'Doc')}] {n.text}" for n in all_nodes[:3]])
            answer = f"Based on retrieved context:\n\n{context_str}\n\nSummary answer to '{query_str}'."
        else:
            answer = f"No relevant context found in index for: '{query_str}'."

        return RaggyResponse(
            response=answer,
            source_nodes=all_nodes,
            confidence_score=confidence,
            reflection_count=iterations,
            metadata={"final_query": current_query},
        )
