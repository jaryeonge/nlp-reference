# Retrieval-Augmented Generation for AI-Generated Content: A Survey

## Abstact
RAG for AIGC에 대한 foundations, enhancements, applications를 다루고, 한계점과 future work를 제시하는 Paper

## 1. Introduction
최근 몇년간 AIGC, Artificial Intelligence Generated Content 영역에서 활발한 연구가 이루어지고 있다. 
AIGC는 사람이 직접하거나 rule-based로 하는 것보다 더 적절한 콘텐츠를 제공하는 것을 의미하며 LLM의 발전으로 인하여 보다 많은 관심을 받고 있다.
그럼에도 불구하고 시스템적인 foundations, enhancements, applications을 언급한 paper가 없어 본 survey를 통해 다루고자 한다.

<img width="687" alt="image" src="https://github.com/jaryeonge/nlp-reference/assets/52644464/3a964523-7853-4aad-a135-c78cad09d031">

## 2. Preliminary
### RAG 2 core modules
1. Retriever: 데이터 저장소에서 관련 있는 정보를 추출하는 모듈
2. Generator: 해당 정보를 기반하여 컨텐츠를 생성하는 모듈

### Generator의 종류
1. Transformer Model: Self-attention - feed forward Network - layer normalization module - residual network로 구성된 NLP 계의 혁신적인 모델
2. LSTM: RNN의 exploding/vanishing gradient 문제를 해결하여 long-term information을 기억할 수 있는 모델
3. Diffusion Model: Forward Process에서 noise를 더하고 reverse process에서 de-noising을 하는 모델이며 현실적이고 다양한 샘플 data를 만들 수 있다.
4. GAN: Real data와 fake data를 generator와 discriminator를 통해 학습하여 만든 generator 영역의 높은 성능을 보이는 모델

### Retriever의 종류
1. Sparse Retriever: Key-value 방식으로 document를 저장하고 TF-IDF, BM25와 같은 방식으로 검색하는 방식
2. Dense Retirever: Sparse와 다르게 embedding vectors를 이용하여 document를 추출하는 방식

## 3. Methods

### RAG Foundations

<img width="946" alt="image" src="https://github.com/jaryeonge/nlp-reference/assets/52644464/6a86e1e1-279e-486b-a0a3-5acd9e9927d7">

1. Query-based RAG: Prompt augmentation이라고도 불리며 retriever를 통해 추출한 document를 generator의 input으로 직접 넣는 방식
** Have to read: REALM, KILT, SELF-RAG, REPLUG
2. Latent Representation-based RAG: Generator와 retrieved objects와 interact하여 최종 output을 산출하는 방식
** Have to read: FiD, Retro, TOME, EaE
3. Logit-based RAG: Retriever와 generator가 독립적으로 output을 산출하여 logit기반으로 결합하는 방식
** Have to read: kNN-LM, TRIME
4. Speculative RAG: Generator를 생략하고 retriever만 사용해서 resouces를 절약하는 방식
** Have to read: REST, GPTCache

### RAG Enhancements

<img width="935" alt="image" src="https://github.com/jaryeonge/nlp-reference/assets/52644464/9ea66156-0d1e-47ea-809c-189fa1464016">

1. Input Enhacement: User query의 quality를 높이는 방식
  a. query transformation: query를 가공하여 더 풍부한 relevant information을 얻을 수 있다.
  b. data augmentation: irrelevant information을 지우고 outdated된 document를 update하는 등의 작업을 통해 retriever의 성능을 향상
2. Retriever Enhancement: Retriever의 quality를 높이는 방식
  a. Recurive Retireve: retrieve 전에 query를 분리하여 multiple search를 하는 방식. CoT의 input으로 이용하기에 적절
  b. Chunk Optimization: chunk의 크기를 조정하여 성능을 향상
  c. Finetune Retriever: retiever를 fine-tuning하여 성능을 향상. embedding model을 tuning하거나 REPLUG처럼 아예 retriever 자체를 훈련할 수도 있다.
  d. Hybrid Retrieve: dense + sparse retrieval methods를 동시에 사용하여 성능을 향상
  e. Re-ranking: retriever의 결과물의 ranking을 조절하여 성능을 향상
  f. Meta-data Filtering: document를 filtering하여 성능을 향상
3. Generator Enhancement: Generator의 quality를 높이는 방식
  a. Prompt Engineering: Stepback Prompt, Active Propmt, Chain of Thought Prompt 등의 prompt 고도화 기법으로 성능을 향상
  b. Decoding Tuning: 별도의 decoder를 추가, 이를 tuning하여 성능을 향상
  c. Finetune Generator: geneator를 fint-tuning하여 성능을 향상
4. Result Enhancement
  a. Rewrite Output: output을 재생성하여 성능을 향상
5. RAG Pipeline Enhancement
  a. Adaptive Retrieval: 의도 분류와 분기처리를 통해 효율적인 리소스 사용, Rule-based & Model-based 방식 존재.
  b. Iterative RAG: 반복적으로 RAG 과정을 수행하여 성능을 향상. query -> retreiver -> generator -> output -> retreiver -> generator ...

## 4. Applications

<img width="932" alt="image" src="https://github.com/jaryeonge/nlp-reference/assets/52644464/df2d845a-6bac-4793-9efd-d2ebb22ad1e8">

1. RAG for Text
  a. Question Answering: 광범위한 답변 후보의 범위를 축소
  b. Fact Verification: Fact information을 활용하여 hallucination 감소
  c. Commonsense Reasoning: commonsense knowledge를 활용하여 human-like한 방식으로 추론하고 decision-making
  d. Human-Machine Conversation: commonsense knowledge를 활용하여 인간과 기계사이의 대화를 끊김없이 제공
  e. Neural Machine Translation: 전통적인 bilingual corpora 의존도를 RAG를 통해 감소
  f. Event Extraction: event의 case를 knowledge로 활용하여 성능 향상
  g. Summarization: non-English 문제를 해결하거나 top-k의 hidden states를 retrieve하여 모델의 longer inputs 처리 능력을 향상
2. RAG for Code: 예제 코드를 retrieve하여 성능 향상
  a. Code Generation
  b. Code Summary
  c. Code Completion
  d. Automatic Program Repair
  e. Text-to-SQL and Code-based Semantic Parsing
3. RAG for Audio
  a. Audio Generation: 관련 오디오를 retrieve하여 성능 향상 및 비슷한 느낌의 오디오를 LLM 학습 없이 생성
  b. Audio Captioning: 관련 캡션을 retrieve하여 성능 향상 및 캡션의 후보를 제시하여 선택할 수 있게 유도
4. RAG for Image
  a. Image Generation: 관련 이미지를 retrieve하여 성능 향상 및 비슷한 느낌의 이미지를 LLM 학습 없이 생성
  b. Image Captioing: 관련 캡션을 retrieve하여 성능 향상 및 캡션의 후보를 제시하여 선택할 수 있게 유도
5. RAG for Video
  a. Video Generation: 관련 plot을 retrieve하여 성능 향상
  b. Video Captioning: 관련 캡션을 retrieve하여 성능 향상 및 캡션의 후보를 제시하여 선택할 수 있게 유도
6. RAG for 3D
  a. Text-to-3D: retrieve한 3D asset을 dffusion 모델에 적용
7. RAG for knowledge
  a. Knowledge Base Question Answering: Knowledege base를 기반하여 답변을 생성하여 domain 최적화
  b. Knowledge Graph Completion: relevant triplets를 retrieve하여 fusion-in-decoder에 적용
8. RAG for Science
  a. Drug Discovery: retrieval 분자와 fusion of exemplar 분자를 input으로 활용
  b. Medical Applications: guidance를 retrieve하여 성능 향상

## 5. Benchmark
### Chen et al.
1. Noise Robustness: input query와 answer의 관련도 측정
2. Neglative Rejection: retreived content가 부족할 때 LLMs가 답변을 거부하는 정도를 측정
3. Information Integration: multiple retrieved content를 통합하는 정도를 측정
4. Counterfactual Robustness: retrieved content 중 counterfactual 항목을 LLMs가 식별하는 정도를 측정

### RAGAS, ARES and TruLens
1. Faithfulness: retrieved content가 정확할 때, factual errors를 내보내는 정도를 측정
2. Answer Relevance: answer가 얼마나 problem solving에 기여하는 지를 측정
3. Context Relevance: retireved content가 얼마나 관련있는 정보를 포함하는 지, 관련없는 정보를 제외하는 지를 측정

### CRUD-RAG
1. CRUD 기반으로 성능을 측정 (자세한 건 Paper 참고)

## 6. Discussion
### Limitations
1. Noises in Retrieval Results: representation의 information loss, ANN의 approximate result로 인한 noise와 필수불가결한 noise 존재. retrieval의 성능과 prompt 개선이 필요
2. Extra Overhead: latency, data storage 증가 문제가 존재, trade-off 고려
3. Interaction of Retireval and Generation: retriever와 generator의 interaction 방법이 최적화되어 있지 않음. 아직까지 적절한 방법론이 제기되지 않았음.
4. Long Context Generation: prompt augmentation의 크기가 너무 커지는 문제가 존재. simple한 prompt로 성능을 끌어올리는 방법론이 필요

### Potential Future Directions
1. More Advanced Research on RAG Methodologies, Enhancements and Appplications: 세 가지 분야에서 더 많은 연구가 필요
2. Efficient Deployment and Processing: LangChain, LLAMA-Index를 제외한 나머지 방법론은 deployment와 processing애 대한 연구가 적음
3. Incorporating Long-tail and Real-time Knowledge: real-time knowledge 업데이트와 long-tail knowledge를 서비스로 연결하기 위한 방법론 연구가 필요
4. Combined with Other Techniques: 현재 이루어지고 있는 연구 분야 외에 다른 곳도 연구가 필요

## 7. Conclusion
RAG의 foundations를 요약하고 이를 기반한 다양한 개선점과 적용안을 제시하였다. 마지막으로 한계점과 future work를 제시하여 RAG 발전의 초석이 되는 Paper로 보인다.
