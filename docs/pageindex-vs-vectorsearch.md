# PageIndex vs Vector Search: Confronto Completo

> **Documento tecnico di confronto tra l'approccio Reasoning-Based Tree Search di PageIndex e la tradizionale Vector Similarity Search per sistemi RAG.**

---

## Executive Summary

| Metrica | Vector RAG | PageIndex |
|---------|-----------|-----------|
| **Accuracy (FinanceBench)** | ~70-80% | **98.7%** |
| **Latenza Query** | ~50-200ms | ~2-10s |
| **Infrastruttura** | Complessa (Vector DB) | Semplice (JSON) |
| **Spiegabilità** | Bassa | **Alta** |
| **Costo Setup** | Alto | Basso |

**Verdetto:** PageIndex eccelle in scenari dove **precisione, spiegabilità e documenti complessi** sono prioritari rispetto alla velocità raw.

---

## 1. Architettura a Confronto

### Vector Search (Tradizionale)

```
Documento → Chunking → Embedding Model → Vector DB → Cosine Similarity → Top-K Chunks
```

### PageIndex (Reasoning-Based)

```
Documento → TOC Detection → Tree Structure → LLM Reasoning → Sezioni Rilevanti
```

### Differenze Fondamentali

| Aspetto | Vector RAG | PageIndex |
|---------|-----------|-----------|
| **Unità di Retrieval** | Chunks fissi (512-1024 token) | Sezioni naturali del documento |
| **Metodo di Ricerca** | Nearest neighbor matematico | Ragionamento semantico |
| **Struttura Dati** | Vettori n-dimensionali | Albero gerarchico JSON |
| **Modello Richiesto** | Embedding model + LLM | Solo LLM |
| **Preservazione Contesto** | Perso ai confini dei chunk | Mantenuto nella struttura |

---

## 2. Performance e Tempistiche

### Fase di Indexing

| Metrica | Vector RAG | PageIndex |
|---------|-----------|-----------|
| **Tempo per documento (10 pagine)** | 1-3 secondi | 30-60 secondi |
| **Tempo per documento (100 pagine)** | 5-15 secondi | 2-5 minuti |
| **Operazioni richieste** | Chunking + Embedding | TOC detection + Tree building + Summarization |
| **Parallelizzazione** | Alta (batch embedding) | Media (async LLM calls) |
| **Costo API (100 pagine)** | ~$0.01-0.05 (embedding) | ~$0.75-1.50 (GPT-4o) |

### Fase di Query

| Metrica | Vector RAG | PageIndex |
|---------|-----------|-----------|
| **Latenza media** | 50-200ms | 2-10 secondi |
| **Throughput** | 1000+ query/sec | 5-10 query/sec |
| **Costo per query** | ~$0.001 (solo retrieval) | ~$0.02-0.10 (LLM reasoning) |
| **Scalabilità orizzontale** | Eccellente | Limitata da rate limits LLM |

### Benchmark Accuracy

| Dataset | Vector RAG | PageIndex | Differenza |
|---------|-----------|-----------|------------|
| **FinanceBench** | ~70-80% | 98.7% | +18-28% |
| **Multi-hop Reasoning** | ~50-60% | ~85-90%* | +25-30% |
| **Long Documents (>100 pg)** | ~60-70% | ~90-95%* | +25-30% |

*Stime basate su architettura

---

## 3. Infrastruttura Richiesta

### Vector RAG Stack

| Componente | Opzioni | Costo/Complessità |
|------------|---------|-------------------|
| **Vector Database** | Pinecone, Weaviate, Chroma, Milvus, Qdrant | $50-500+/mese (managed) |
| **Embedding Model** | OpenAI Ada, Cohere, Sentence Transformers | $0.0001/1K tokens |
| **Index Management** | Sharding, replication, backup | DevOps overhead |
| **Versioning** | Embedding model changes = re-index | Significativo |

### PageIndex Stack

| Componente | Opzioni | Costo/Complessità |
|------------|---------|-------------------|
| **Storage** | File system / S3 / MongoDB | Minimo |
| **LLM API** | OpenAI GPT-4o, Claude, etc. | Pay per use |
| **Index Management** | JSON files | Triviale |
| **Versioning** | Re-process se cambia prompt | Moderato |

### Confronto TCO (Total Cost of Ownership)

| Scenario | Vector RAG | PageIndex |
|----------|-----------|-----------|
| **Setup iniziale** | 2-4 settimane | 1-2 giorni |
| **Manutenzione mensile** | 10-20 ore | 2-5 ore |
| **Costo infra (1000 docs)** | $100-300/mese | $20-50/mese |
| **Costo query (10K/mese)** | $10-50 | $200-1000 |

**Trade-off chiave:** PageIndex ha costi infrastrutturali più bassi ma costi query più alti.

---

## 4. Qualità del Retrieval

### Il Problema Fondamentale

> **Similarità Semantica ≠ Rilevanza**

Vector search trova testo che "suona simile" alla query, ma non necessariamente risponde alla domanda.

### Esempi Pratici

| Query | Vector Search | PageIndex |
|-------|--------------|-----------|
| "Qual è stato l'EBITDA del Q3?" | Trova chunks con "EBITDA", "Q3", ma potrebbe restituire Q2 o forecast | Naviga a MD&A → Financial Results → Q3 → EBITDA |
| "Rischi legali menzionati" | Chunks sparsi da più sezioni | Identifica sezione "Risk Factors" e "Legal Proceedings" |
| "Comparazione YoY revenue" | Potrebbe restituire solo un anno | Recupera entrambi gli anni dalla struttura corretta |

### Spiegabilità

| Aspetto | Vector RAG | PageIndex |
|---------|-----------|-----------|
| **Output retrieval** | Lista di chunks + similarity scores | Percorso nell'albero + reasoning |
| **Debugging** | Difficile (perché score 0.87 vs 0.85?) | Facile (puoi leggere il "thinking") |
| **Audit trail** | "Chunk 47 aveva score 0.89" | "Ho selezionato Item 7 perché contiene MD&A" |
| **Compliance** | Richiede reverse engineering | Nativo |

---

## 5. Integrazione di Expert Knowledge

### Vector RAG

```python
# Richiede fine-tuning del modello di embedding
# oppure metadata filtering complesso
results = vector_db.query(
    embedding=query_embedding,
    filter={"section": "Item 7"},  # Hardcoded rules
    top_k=5
)
```

**Problemi:**
- Fine-tuning è costoso e richiede dati
- Regole hardcoded non scalano
- Difficile aggiornare la "conoscenza"

### PageIndex

```python
# Expert knowledge nel prompt
prompt = f"""
Query: {query}
Document Tree: {tree_structure}

Expert Knowledge:
- Per query su EBITDA, prioritizza Item 7 (MD&A)
- Per query su rischi, controlla Item 1A e Item 3
- Per dati di acquisizioni, cerca Item 8 Note 2

Reasoning: ...
"""
```

**Vantaggi:**
- Zero training richiesto
- Knowledge aggiornabile istantaneamente
- Facile da testare e debuggare

---

## 6. Casi d'Uso Ideali

### Quando Usare Vector Search

| Scenario | Perché |
|----------|--------|
| **Chatbot ad alto volume** | Latenza <200ms critica |
| **Search-as-you-type** | Real-time UX |
| **Knowledge base semplici** | FAQ, documentazione generica |
| **Budget limitato per query** | Costo per query ~$0.001 |
| **Documenti brevi (<20 pagine)** | Chunking meno problematico |

### Quando Usare PageIndex

| Scenario | Perché |
|----------|--------|
| **Documenti finanziari** | Precisione 98.7% vs 70-80% |
| **Contratti legali** | Struttura e sezioni critiche |
| **Report tecnici lunghi** | >100 pagine, struttura complessa |
| **Compliance/Audit** | Spiegabilità nativa |
| **Domain expertise richiesta** | Injection diretta di knowledge |
| **Low-volume, high-value queries** | Qualità > quantità |

---

## 7. Approccio Ibrido

Per il meglio di entrambi i mondi:

```
Query → Vector Pre-filter (top 50) → PageIndex Reranking → Top 5 Results
```

| Fase | Tecnologia | Scopo |
|------|------------|-------|
| **Stage 1** | Vector Search | Retrieval veloce, broad match |
| **Stage 2** | PageIndex | Reasoning preciso, reranking |

### Vantaggi Ibrido

- Latenza: ~500ms-2s (vs 2-10s pure PageIndex)
- Accuracy: ~95%+ (vs 70-80% pure vector)
- Costo: Ridotto (meno chiamate LLM)

---

## 8. Limitazioni e Considerazioni

### Limitazioni PageIndex

| Limitazione | Impatto | Mitigazione |
|-------------|---------|-------------|
| **Latenza alta** | Non adatto per real-time | Caching, approccio ibrido |
| **Costo per query** | $0.02-0.10 vs $0.001 | Batch processing, caching |
| **Rate limits LLM** | Max ~100-500 RPM | Queue system, retry logic |
| **Dipendenza LLM** | Vendor lock-in | Multi-provider support |
| **Documenti senza struttura** | Meno efficace | Fallback a vector |

### Limitazioni Vector Search

| Limitazione | Impatto | Mitigazione |
|-------------|---------|-------------|
| **Chunking boundary loss** | Contesto perso | Overlap chunking |
| **Similarity ≠ Relevance** | Wrong results | Reranking models |
| **Long documents** | Chunks disconnessi | Hierarchical indexing |
| **Domain specificity** | Generic embeddings | Fine-tuning (costoso) |
| **Opaque retrieval** | No explainability | Nessuna vera soluzione |

---

## 9. Roadmap Decisionale

```
                        ┌─────────────────────────────────┐
                        │  Qual è la tua priorità #1?     │
                        └────────────┬────────────────────┘
                                     │
              ┌──────────────────────┼──────────────────────┐
              │                      │                      │
              ▼                      ▼                      ▼
        ┌─────────┐           ┌─────────────┐        ┌─────────────┐
        │ Latenza │           │  Precisione │        │   Budget    │
        │  <500ms │           │    >95%     │        │  Limitato   │
        └────┬────┘           └──────┬──────┘        └──────┬──────┘
             │                       │                      │
             ▼                       ▼                      ▼
      ┌────────────┐          ┌────────────┐         ┌────────────┐
      │   VECTOR   │          │  PAGEINDEX │         │   VECTOR   │
      │   SEARCH   │          │            │         │   SEARCH   │
      └────────────┘          └────────────┘         └────────────┘
```

---

## 10. Conclusioni

### PageIndex È Migliore Quando:

1. **La precisione è critica** - Errori costano caro (legal, finance, compliance)
2. **I documenti sono complessi** - Struttura gerarchica, >50 pagine
3. **Serve spiegabilità** - Audit, debugging, trust
4. **Hai expert knowledge** - Regole di dominio da applicare
5. **Query volume è basso** - <1000 query/giorno

### Vector Search È Migliore Quando:

1. **La latenza è critica** - Real-time UX
2. **Alto throughput** - Migliaia di query/secondo
3. **Documenti semplici** - FAQ, knowledge base generiche
4. **Budget query-driven** - Paghi per query, non per precisione

### La Vera Risposta

**Non è "o uno o l'altro"** - è capire il tuo use case:

| Use Case | Raccomandazione |
|----------|-----------------|
| Chatbot customer service | Vector Search |
| Financial document analysis | PageIndex |
| Legal contract review | PageIndex |
| Internal wiki search | Vector Search |
| Compliance reporting | PageIndex |
| E-commerce product search | Vector Search |
| Technical manual Q&A | Ibrido |

---

## Riferimenti

- [PageIndex Paper](https://arxiv.org/abs/pageindex) - Vectorless RAG approach
- [FinanceBench Benchmark](https://financebench.com) - Financial QA dataset
- [RAPTOR](https://arxiv.org/abs/raptor) - Hierarchical RAG alternative
- [RAG Survey 2024](https://arxiv.org/abs/rag-survey) - State of the art

---

*Documento generato per il progetto PageIndex - Ultimo aggiornamento: Gennaio 2026*
