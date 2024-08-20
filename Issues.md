# GG VectorDB Integration: Issues and Suggestions

## A. Issues

### 1. Batch Update Failures
- **Issue**: Batch updates fail for a small number of records.
- **Suggestion**: The best would be to have an exception or some documentation in this regard would be helpful

**Current Implementation:**
```java
while ((line = reader.readNext()) != null) {
    lineCnt++;
    System.out.println("line[0]: "+line[0]);
    Long key = Long.valueOf(line[0]);
    Article val = new Article(line[1], line[2], line[3]);

    batch.put(key, val);

    if (lineCnt % batchSz == 0) {
        articleCache.putAll(batch);
        batch.clear();
        System.out.println(lineCnt + " docs indexed ...");
    }
}
```

### 2. Exact Text Matching Requirement
- **Issue**: The system requires the exact text from the CSV to be sent, making the process error-prone. When CSV text has no space, but the added data does, it directly results in a null vector issue.
- **Consequences**:
  - This is very difficult to understand and debug.
  - The entire database becomes unstable once this error is encountered.
- **Suggestions**: 
  - Provide clearer error messages and documentation on text formatting requirements.

**Example Exception:**
```java
java.lang.IllegalArgumentException: vector value must not be null
    at org.apache.lucene.document.KnnFloatVectorField.createType(KnnFloatVectorField.java:44)
    at org.apache.lucene.document.KnnFloatVectorField.<init>(KnnFloatVectorField.java:100)
    at org.gridgain.internal.processors.query.h2.opt.GridLuceneIndex.store(GridLuceneIndex.java:211)
    // ... (stack trace truncated for brevity)
```

### 3. Issue in the Example Code
- **Issue**: The provided example code includes a `Person` class, which is confusing in the context of article indexing.
- **Suggestion**: Update example code to use more relevant and consistent class names (e.g., `Article` instead of `Person`).

**Current Implementation:**
```java
private static void vectorQuery() throws IOException {
    IgniteCache<Long, Person> cache = Ignition.ignite().cache(ARTICLE_CACHE);

    String query = resolveIgnitePath("examples/vector/query-full.txt").getAbsolutePath();
    // Vector for "Is the Atlantic the biggest ocean in the world?" as string.
    String queryVector = U.readFileToString(query, "UTF-8");
    System.out.println("queryVector is "+ queryVector);

    // Execute queries for salary ranges.
    print("Query the Lucene index: ",
        cache.query(new VectorQuery(Article.class, "content", toVector(queryVector))).getAll());
}
```

## B. Enhancements

### 1. Embedding Management
- **Issue**: The current approach of adding articles without embeddings and relying on a backend translator to manage embeddings seems roundabout.
- **Suggestion**: Consider implementing a more direct approach to embedding management.

#### API Design for Vector Embeddings
- **Observation**: Most vector databases provide APIs for directly sending vector embeddings or accepting API keys for embedding model providers.
- **Suggestion**: Consider implementing similar API designs for more flexible and efficient embedding management.

**Comparison: AstraDB Vector Store Implementation**

[Reference link](https://github.com/langchain-ai/langchain/blob/master/docs/docs/integrations/vectorstores/astradb.ipynb)

1. Explicit Embedding Initialization:

```python
from langchain_openai import OpenAIEmbeddings
from langchain_astradb import AstraDBVectorStore

embeddings = OpenAIEmbeddings(model="text-embedding-3-large")

vector_store = AstraDBVectorStore(
  collection_name="astra_vector_langchain",
  embedding=embeddings,
  api_endpoint=ASTRA_DB_API_ENDPOINT,
  token=ASTRA_DB_APPLICATION_TOKEN,
  namespace=ASTRA_DB_NAMESPACE,
)
```

2. Integrated Embeddings Initialization:

```python
from astrapy.info import CollectionVectorServiceOptions

openai_vectorize_options = CollectionVectorServiceOptions(
  provider="openai",
  model_name="text-embedding-3-small",
  authentication={
    "providerKey": "OPENAI_API_KEY",
  },
)

vector_store_integrated = AstraDBVectorStore(
  collection_name="astra_vector_langchain_integrated",
  api_endpoint=ASTRA_DB_API_ENDPOINT,
  token=ASTRA_DB_APPLICATION_TOKEN,
  namespace=ASTRA_DB_NAMESPACE,
  collection_vector_service_options=openai_vectorize_options,
)
```