"""
Graph RAG — Local Knowledge Graph with Embeddings + Ollama
"""

import json, re
import networkx as nx
import numpy as np
import ollama
from sentence_transformers import SentenceTransformer

# ── Config ─────────────────────────────────────────────────
MODEL = "gemma3:1b"
GRAPH_FILE = "knowledge_graph.json"
embedder = SentenceTransformer("all-MiniLM-L6-v2")


def llm(system: str, user: str) -> str:
    """Chat with local Ollama."""
    r = ollama.chat(model=MODEL, messages=[
        {"role": "system", "content": system},
        {"role": "user", "content": user},
    ])
    return r["message"]["content"].strip()


# ── Source Index ───────────────────────────────────────────

class SourceIndex:
    """Chunks source text, maps entities back to passages."""

    def __init__(self):
        self.chunks = []

    def load(self, text: str, chunk_size=800, overlap=200):
        self.chunks = []
        cid = 0
        for section in re.split(r'(━{10,})', text):
            section = section.strip()
            if not section or set(section) == {'━'}:
                continue
            mrn = (re.search(r'MRN-[\d-]+', section) or type('', (), {'group': lambda s: 'cross-ref'})()).group()
            if len(section) <= chunk_size:
                self.chunks.append({"id": cid, "text": section, "mrn": mrn})
                cid += 1
            else:
                for start in range(0, len(section), chunk_size - overlap):
                    chunk = section[start:start + chunk_size]
                    if len(chunk.strip()) > 50:
                        self.chunks.append({"id": cid, "text": chunk, "mrn": mrn})
                        cid += 1
        print(f"  Indexed {len(self.chunks)} chunks")

    def find(self, entity_text: str) -> list[int]:
        t = entity_text.lower()
        return [c["id"] for c in self.chunks if t in c["text"].lower()]

    def get(self, ids: list[int]) -> list[dict]:
        return [c for c in self.chunks if c["id"] in ids]


# ── Knowledge Graph ───────────────────────────────────────

class KnowledgeGraph:
    def __init__(self):
        self.graph = nx.DiGraph()
        self.source = SourceIndex()
        self._texts = []
        self._embeds = None

    def ingest(self, text: str):
        self.source.load(text)

    def add_entities(self, entities: list[dict]):
        for e in entities:
            nid = f"{e['class']}::{e['text']}"
            chunks = self.source.find(e["text"])
            attrs = " ".join(f"{k}={v}" for k, v in e["attributes"].items())
            full = f"{e['class']} {e['text']} {attrs}"
            self.graph.add_node(nid, cls=e["class"], text=e["text"],
                                attrs=e["attributes"], full=full, chunks=chunks)
            self._texts.append(full)

        self._embeds = embedder.encode(self._texts, show_progress_bar=False)
        linked = sum(1 for _, d in self.graph.nodes(data=True) if d.get("chunks"))
        print(f"  {len(entities)} nodes ({linked} linked), embeddings: {self._embeds.shape}")

    def infer_relationships(self, batch_size=15):
        nodes = list(self.graph.nodes(data=True))
        groups = {}
        for nid, d in nodes:
            groups.setdefault(d["cls"], []).append((nid, d))

        pairs = [
            ("patient","provider"), ("patient","diagnosis"), ("patient","medication"),
            ("patient","procedure"), ("patient","lab_result"), ("patient","facility"),
            ("patient","research"), ("patient","device"), ("patient","follow_up"),
            ("diagnosis","medication"), ("diagnosis","procedure"), ("diagnosis","diagnosis"),
            ("provider","procedure"), ("provider","facility"), ("provider","provider"),
            ("provider","research"), ("medication","medication"), ("procedure","device"),
            ("assessment_score","diagnosis"), ("patient","assessment_score"),
        ]

        total = 0
        for ca, cb in pairs:
            ga, gb = groups.get(ca, []), groups.get(cb, [])
            if not ga or not gb:
                continue
            ps = [(a, b) for a in ga for b in gb if a[0] != b[0]] if ca == cb else [(a, b) for a in ga for b in gb]

            for i in range(0, len(ps), batch_size):
                batch = ps[i:i + batch_size]
                desc = "\n".join(
                    f"  {j+1}. [{a[1]['cls']}] \"{a[1]['text']}\" → [{b[1]['cls']}] \"{b[1]['text']}\" "
                    f"({'SAME' if set(a[1].get('chunks',[])) & set(b[1].get('chunks',[])) else 'DIFF'})"
                    for j, (a, b) in enumerate(batch)
                )
                try:
                    resp = llm(
                        "Medical KG expert. For each pair: <num>|YES/NO|RELATIONSHIP|desc",
                        f"Entity pairs:\n{desc}\n\nOne line per pair."
                    )
                    for line in resp.strip().split("\n"):
                        if "|" not in line: continue
                        p = line.split("|")
                        if len(p) < 3: continue
                        try:
                            idx = int(p[0].strip().rstrip(".")) - 1
                            if p[1].strip().upper() == "YES" and 0 <= idx < len(batch):
                                self.graph.add_edge(batch[idx][0][0], batch[idx][1][0],
                                    rel=p[2].strip().upper(), desc=p[3].strip() if len(p) > 3 else "")
                                total += 1
                        except (ValueError, IndexError): pass
                except Exception as e:
                    print(f"    Warn: {e}")

        # Co-occurrence edges
        ns = list(self.graph.nodes(data=True))
        for i, (a, da) in enumerate(ns):
            ca = set(da.get("chunks", []))
            if not ca: continue
            for j, (b, db) in enumerate(ns):
                if j <= i: continue
                cb = set(db.get("chunks", []))
                if ca & cb and da["cls"] != db["cls"] and not self.graph.has_edge(a, b) and not self.graph.has_edge(b, a):
                    self.graph.add_edge(a, b, rel="CO_OCCURS", desc=f"{len(ca & cb)} shared chunks")
                    total += 1

        print(f"  {self.graph.number_of_nodes()} nodes, {self.graph.number_of_edges()} edges")

    def query(self, question: str, top_k=10) -> str:
        if self._embeds is None: return "No graph loaded."

        # Semantic search
        qe = embedder.encode([question])
        qn = qe / np.linalg.norm(qe, axis=1, keepdims=True)
        nn = self._embeds / np.linalg.norm(self._embeds, axis=1, keepdims=True)
        sims = (qn @ nn.T).flatten()
        top = np.argsort(sims)[::-1][:top_k]
        nids = list(self.graph.nodes())
        relevant = [(nids[i], float(sims[i])) for i in top if sims[i] > 0.1]

        if not relevant: return "No relevant entities found."

        # Expand 1-hop
        sub = set()
        for nid, _ in relevant:
            sub.add(nid)
            sub.update(self.graph.successors(nid))
            sub.update(self.graph.predecessors(nid))

        # Collect source chunks
        chunk_ids = set()
        for nid in sub:
            if nid in self.graph:
                chunk_ids.update(self.graph.nodes[nid].get("chunks", []))
        passages = self.source.get(list(chunk_ids))

        # Build context
        ctx = ["=== ENTITIES ==="]
        for nid in sub:
            if nid not in self.graph: continue
            d = self.graph.nodes[nid]
            s = next((sc for n, sc in relevant if n == nid), None)
            ctx.append(f"[{d['cls']}] \"{d['text']}\"" + (f" (sim:{s:.2f})" if s else ""))

        ctx.append("\n=== RELATIONSHIPS ===")
        for u, v, d in self.graph.edges(data=True):
            if u in sub and v in sub:
                ctx.append(f"\"{self.graph.nodes[u]['text']}\" --[{d.get('rel','')}]--> \"{self.graph.nodes[v]['text']}\"")

        ctx.append("\n=== SOURCE TEXT ===")
        for p in passages[:6]:
            ctx.append(f"\n[Chunk {p['id']}, {p['mrn']}]\n{p['text'][:500]}")

        return llm(
            "Medical assistant. Answer using the knowledge graph and source text. Be precise.",
            "\n".join(ctx) + f"\n\nQuestion: {question}"
        )

    def save(self, fp=GRAPH_FILE):
        json.dump({"graph": nx.node_link_data(self.graph), "chunks": self.source.chunks},
                  open(fp, "w", encoding="utf-8"), indent=2, ensure_ascii=False)
        print(f"  Saved to {fp}")

    def load(self, fp=GRAPH_FILE):
        d = json.load(open(fp, "r", encoding="utf-8"))
        self.graph = nx.node_link_graph(d["graph"])
        self.source.chunks = d.get("chunks", [])
        self._texts = [self.graph.nodes[n].get("full", "") for n in self.graph.nodes()]
        if self._texts:
            self._embeds = embedder.encode(self._texts, show_progress_bar=False)
        print(f"  Loaded: {self.graph.number_of_nodes()} nodes, {self.graph.number_of_edges()} edges")

    def stats(self):
        by_cls = {}
        for _, d in self.graph.nodes(data=True):
            by_cls[d.get("cls", "?")] = by_cls.get(d.get("cls", "?"), 0) + 1
        by_rel = {}
        for _, _, d in self.graph.edges(data=True):
            by_rel[d.get("rel", "?")] = by_rel.get(d.get("rel", "?"), 0) + 1
        print(f"\n  Nodes: {self.graph.number_of_nodes()} | Edges: {self.graph.number_of_edges()}")
        print(f"  Chunks: {len(self.source.chunks)}")
        for c, n in sorted(by_cls.items(), key=lambda x: -x[1]): print(f"    {c}: {n}")
        for r, n in sorted(by_rel.items(), key=lambda x: -x[1]): print(f"    {r}: {n}")

    def visualize(self, output="graph.html"):
        """Generate interactive HTML graph visualization."""
        from pyvis.network import Network
        colors = {
            "patient": "#e74c3c", "provider": "#3498db", "diagnosis": "#e67e22",
            "medication": "#2ecc71", "procedure": "#9b59b6", "lab_result": "#1abc9c",
            "facility": "#f39c12", "device": "#34495e", "research": "#e91e63",
            "assessment_score": "#00bcd4", "follow_up": "#8bc34a",
        }
        net = Network(height="800px", width="100%", directed=True, bgcolor="#1a1a2e", font_color="white")
        net.barnes_hut(gravity=-3000, spring_length=150)

        for nid, d in self.graph.nodes(data=True):
            attrs = ", ".join(f"{k}={v}" for k, v in d.get("attrs", {}).items())
            net.add_node(nid, label=d["text"], color=colors.get(d["cls"], "#95a5a6"),
                         title=f"[{d['cls']}]\n{attrs}", size=20 + len(d.get("chunks", [])) * 3)

        for u, v, d in self.graph.edges(data=True):
            net.add_edge(u, v, title=d.get("rel", ""), label=d.get("rel", ""),
                         color="#ffffff55", arrows="to")

        net.write_html(output)
        print(f"  Graph visualization: {output} (open in browser)")


if __name__ == "__main__":
    kg = KnowledgeGraph()
    kg.ingest(open("sample_data.txt", "r", encoding="utf-8").read())
    entities = json.load(open("entities.json", "r", encoding="utf-8"))
    kg.add_entities(entities)
    kg.infer_relationships()
    kg.stats()
    kg.save()
    kg.visualize()
    print("\n" + kg.query("What medications is Rajesh Venkataraman taking and why?"))
