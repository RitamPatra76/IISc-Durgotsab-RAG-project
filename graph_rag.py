"""
IISc Sharodiya Durgotsab — Graph RAG Retriever & Question Answering System
Retrieves information from the knowledge graph (Neo4j or NetworkX) using
keyword matching and graph traversal, then generates grounded answers using an LLM.
"""

import json
import os
import re
import sys
from typing import Optional, List, Dict, Any

# Ensure UTF-8 output on Windows console for Bengali text
if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
        sys.stderr.reconfigure(encoding="utf-8")
    except Exception:
        pass

from dotenv import load_dotenv

# Search for .env in current dir and parent directories
load_dotenv()
load_dotenv(os.path.join(os.path.dirname(__file__), "..", ".env"))

try:
    from neo4j import GraphDatabase
    HAS_NEO4J = True
except ImportError:
    HAS_NEO4J = False

import networkx as nx

try:
    import openai
    HAS_OPENAI = True
except ImportError:
    HAS_OPENAI = False


class GraphRAGRetriever:
    """
    Graph-based retrieval engine for RAG.
    Supports Neo4j and NetworkX backends with neighborhood expansion.
    """

    def __init__(
        self,
        backend: str = "networkx",
        neo4j_uri: Optional[str] = None,
        neo4j_user: Optional[str] = None,
        neo4j_password: Optional[str] = None,
        graph_path: Optional[str] = None,
    ):
        self.backend = backend
        self.G = None
        self.driver = None

        if backend == "neo4j" and HAS_NEO4J and neo4j_uri:
            self.driver = GraphDatabase.driver(neo4j_uri, auth=(neo4j_user, neo4j_password))
        else:
            self.backend = "networkx"
            if not graph_path:
                project_dir = os.path.dirname(os.path.abspath(__file__))
                for candidate in ["graph.json", "graph.graphml"]:
                    p = os.path.join(project_dir, candidate)
                    if os.path.exists(p):
                        graph_path = p
                        break

            if graph_path and os.path.exists(graph_path):
                if graph_path.endswith(".json"):
                    from networkx.readwrite import json_graph
                    with open(graph_path, "r", encoding="utf-8") as f:
                        data = json.load(f)
                    self.G = json_graph.node_link_graph(data)
                else:
                    self.G = nx.read_graphml(graph_path)
                print(f"  Loaded graph: {self.G.number_of_nodes()} nodes, {self.G.number_of_edges()} edges")
            else:
                raise FileNotFoundError(f"Graph file not found: {graph_path}")

    def close(self):
        if self.driver:
            self.driver.close()

    def search_nodes(self, query: str, node_types: Optional[List[str]] = None, limit: int = 12) -> List[Dict[str, Any]]:
        """Search nodes by keyword matching across all node attributes."""
        if self.backend == "neo4j":
            return self._neo4j_search(query, node_types, limit)
        else:
            return self._networkx_search(query, node_types, limit)

    def get_node_context(self, node_id: str, hops: int = 1) -> Dict[str, Any]:
        """Get a node and its neighborhood up to N hops."""
        if self.backend == "neo4j":
            return self._neo4j_context(node_id, hops)
        else:
            return self._networkx_context(node_id, hops)

    def get_by_type(self, node_type: str) -> List[Dict[str, Any]]:
        """Get all nodes of a given entity type."""
        if self.backend == "neo4j":
            return self._neo4j_by_type(node_type)
        else:
            return self._networkx_by_type(node_type)

    def retrieve_for_query(self, query: str, max_context_nodes: int = 15) -> str:
        """
        Main retrieval method for RAG.
        Extracts matched nodes, expands 1-hop subgraph relationships, and formats readable context.
        """
        query_lower = query.lower()
        type_hints = self._infer_types(query_lower)

        matches = self.search_nodes(query, node_types=type_hints, limit=max_context_nodes)
        
        # If type hints were detected, backfill nodes of those types if matches are few
        if type_hints and len(matches) < 3:
            existing_ids = {m.get("id") for m in matches}
            for th in type_hints:
                for typed_node in self.get_by_type(th):
                    if typed_node.get("id") not in existing_ids:
                        matches.append(typed_node)
                        existing_ids.add(typed_node.get("id"))
                        if len(matches) >= max_context_nodes:
                            break

        if not matches:
            matches = self.search_nodes(query, limit=max_context_nodes)

        context_parts = []
        seen = set()

        for match in matches:
            node_id = match.get("id", "")
            if not node_id or node_id in seen:
                continue
            seen.add(node_id)
            ctx = self.get_node_context(node_id, hops=1)
            formatted = self._format_context(ctx)
            if formatted:
                context_parts.append(formatted)

        # Include festival/location anchor if asking about schedule or dates
        if any(kw in query_lower for kw in ["when", "date", "time", "where", "location", "address", "gate", "reach"]):
            festival_nodes = self.get_by_type("Festival")
            for fn in festival_nodes[:1]:
                if fn.get("id") not in seen:
                    ctx = self.get_node_context(fn["id"], hops=1)
                    context_parts.insert(0, self._format_context(ctx))

        return "\n\n---\n\n".join(context_parts) if context_parts else "No relevant information found in knowledge graph."

    # ─── NetworkX Search & Subgraph ─────────────────────────────────

    @staticmethod
    def _normalize(t: str) -> str:
        t = t.lower().strip()
        if t.endswith("ies") and len(t) > 4:
            return t[:-3] + "y"
        if t.endswith("es") and len(t) > 4:
            return t[:-2]
        if t.endswith("s") and len(t) > 3:
            return t[:-1]
        return t

    def _networkx_search(self, query: str, node_types: Optional[List[str]] = None, limit: int = 12) -> List[Dict[str, Any]]:
        raw_terms = re.findall(r"\w+", query.lower())
        norm_terms = {self._normalize(t) for t in raw_terms if len(t) > 2 or t in ["tmc", "sbi", "bus", "cab", "upi", "fit", "fee"]}
        results = []

        for node_id, attrs in self.G.nodes(data=True):
            if node_types and attrs.get("type") not in node_types:
                continue

            searchable = " ".join(
                f"{k} {v}".lower() for k, v in attrs.items() if isinstance(v, (str, int, float, list))
            )

            score = 0
            for term in norm_terms:
                if term in searchable:
                    score += 1
                    name_field = str(attrs.get("name") or attrs.get("title") or attrs.get("name_en") or attrs.get("term") or attrs.get("gate_title") or "").lower()
                    if term in name_field:
                        score += 3

            if score > 0:
                results.append({"id": node_id, "score": score, **attrs})

        results.sort(key=lambda x: x["score"], reverse=True)
        return results[:limit]

    def _networkx_context(self, node_id: str, hops: int = 1) -> Dict[str, Any]:
        if not self.G.has_node(node_id):
            return {}

        props = dict(self.G.nodes[node_id])
        neighbors = []

        # Outgoing edges
        for target in self.G.successors(node_id):
            edge_data = self.G.get_edge_data(node_id, target) or {}
            target_data = dict(self.G.nodes[target])
            neighbors.append({
                "direction": "outgoing",
                "relationship": edge_data.get("relationship", "CONNECTED_TO"),
                "node": {"id": target, **target_data},
            })

        # Incoming edges
        for source in self.G.predecessors(node_id):
            edge_data = self.G.get_edge_data(source, node_id) or {}
            source_data = dict(self.G.nodes[source])
            neighbors.append({
                "direction": "incoming",
                "relationship": edge_data.get("relationship", "CONNECTED_TO"),
                "node": {"id": source, **source_data},
            })

        return {"node_id": node_id, "properties": props, "neighbors": neighbors}

    def _networkx_by_type(self, node_type: str) -> List[Dict[str, Any]]:
        results = []
        for node_id, attrs in self.G.nodes(data=True):
            if attrs.get("type") == node_type:
                results.append({"id": node_id, **attrs})
        return results

    # ─── Neo4j Implementation ───────────────────────────────────────

    def _neo4j_search(self, query: str, node_types: Optional[List[str]] = None, limit: int = 12) -> List[Dict[str, Any]]:
        query_words = re.findall(r"\w+", query)
        safe_query = " OR ".join(w for w in query_words if len(w) > 2)

        type_filter = ""
        if node_types:
            type_filter = "WHERE " + " OR ".join(f"n:{nt}" for nt in node_types)

        cypher = f"""
            CALL db.index.fulltext.queryNodes('search_index', $query) YIELD node, score
            {type_filter}
            RETURN labels(node)[0] AS type, properties(node) AS props, score
            ORDER BY score DESC LIMIT $limit
        """
        results = []
        try:
            with self.driver.session() as session:
                records = session.run(cypher, query=safe_query, limit=limit)
                for r in records:
                    props = dict(r["props"])
                    props["type"] = r["type"]
                    props["score"] = r["score"]
                    results.append(props)
        except Exception:
            # Fallback to CONTAINS
            cypher = f"""
                MATCH (n)
                {type_filter}
                WHERE any(prop in keys(n) WHERE toString(n[prop]) =~ '(?i).*' + $q + '.*')
                RETURN labels(n)[0] AS type, properties(n) AS props LIMIT $limit
            """
            with self.driver.session() as session:
                records = session.run(cypher, q=query_words[0] if query_words else "", limit=limit)
                for r in records:
                    props = dict(r["props"])
                    props["type"] = r["type"]
                    results.append(props)
        return results

    def _neo4j_context(self, node_id: str, hops: int = 1) -> Dict[str, Any]:
        cypher = """
            MATCH (n {id: $id})
            OPTIONAL MATCH (n)-[r]->(out)
            OPTIONAL MATCH (inc)-[r2]->(n)
            RETURN properties(n) AS props, labels(n)[0] AS type,
                   collect(DISTINCT {direction: 'outgoing', relationship: type(r), node: properties(out), type: labels(out)[0]}) AS out_neighbors,
                   collect(DISTINCT {direction: 'incoming', relationship: type(r2), node: properties(inc), type: labels(inc)[0]}) AS inc_neighbors
        """
        with self.driver.session() as session:
            record = session.run(cypher, id=node_id).single()
            if not record:
                return {}
            neighbors = []
            for n in record["out_neighbors"]:
                if n.get("relationship"):
                    neighbors.append(n)
            for n in record["inc_neighbors"]:
                if n.get("relationship"):
                    neighbors.append(n)
            props = dict(record["props"])
            props["type"] = record["type"]
            return {"node_id": node_id, "properties": props, "neighbors": neighbors}

    def _neo4j_by_type(self, node_type: str) -> List[Dict[str, Any]]:
        cypher = f"MATCH (n:{node_type}) RETURN properties(n) AS props LIMIT 50"
        results = []
        with self.driver.session() as session:
            for r in session.run(cypher):
                results.append(dict(r["props"]))
        return results

    # ─── Query Routing & Context Formatting ─────────────────────────

    def _infer_types(self, query: str) -> Optional[List[str]]:
        type_map = {
            "FestivalDay": ["schedule", "day", "program", "programme", "when", "date", "timing", "ashtami", "shashthi", "saptami", "nabami", "dashami", "mahalaya"],
            "Ritual": ["anjali", "pushpanjali", "sandhi", "puja", "arati", "sandhya", "bhog", "prasad", "boron", "sindoor khela", "bisarjan", "immersion", "bodhon", "tarpan", "chokkhu"],
            "GateDirection": ["gate", "entrance", "walking", "direction", "how to enter", "main gate", "gymkhana", "d gate", "new bel", "marker", "grid"],
            "TransportOption": ["bus", "train", "railway", "airport", "auto", "taxi", "cab", "metro", "station", "yeshwantpur", "wheelchair", "accessibility", "pram", "reach"],
            "CommitteeMember": ["committee", "convenor", "organizer", "secretary", "treasurer", "phone", "contact person", "advisor", "tathagata", "devraj", "sirshendu", "tapajyoti"],
            "VolunteerTeam": ["volunteer", "join", "team", "decoration", "logistics", "fundraising", "content", "serve", "help"],
            "SponsorshipTier": ["tier", "package", "cost", "diamond", "platinum", "gold", "silver", "bronze", "stall", "banner", "branding", "standee"],
            "Sponsor": ["sponsor", "genotypic", "meme theory", "partner", "dhitiomics", "qtlomics", "funded by"],
            "Merchandise": ["tshirt", "t-shirt", "shirt", "merch", "merchandise", "buy", "purchase", "price", "size", "fit", "petrol blue"],
            "Publication": ["magazine", "probash", "cover", "write", "submit", "poem", "essay", "article", "publish"],
            "SoundscapeLayer": ["music", "song", "dhak", "shankha", "kansar", "sound", "playlist", "audio", "listen"],
            "GalleryAlbum": ["photo", "gallery", "image", "picture", "album", "archive", "photograph"],
            "ArtForm": ["art", "craft", "shilpa", "alpona", "chalchitra", "daker saj", "sholar saj", "putul", "patachitra", "dhak"],
            "HistoricalFact": ["history", "origin", "started", "first", "year", "ancient", "tradition", "1583", "1910"],
            "GlossaryTerm": ["meaning", "what is", "define", "term", "glossary", "word"],
            "Donation": ["donate", "donation", "upi", "pay", "money", "receipt", "contribute", "account"],
        }

        matched = []
        for node_type, keywords in type_map.items():
            if any(kw in query for kw in keywords):
                matched.append(node_type)
        return matched or None

    def _format_context(self, ctx: Dict[str, Any]) -> str:
        if not ctx:
            return ""

        props = ctx.get("properties", {})
        node_type = props.get("type", "Entity")
        lines = [f"[{node_type}]"]

        skip_keys = {"type", "score", "uid", "id"}
        for key, value in props.items():
            if key in skip_keys or not value:
                continue
            clean_key = key.replace("_", " ").title()
            lines.append(f"  {clean_key}: {value}")

        if ctx.get("neighbors"):
            lines.append("  Related Entities:")
            for n in ctx["neighbors"][:8]:
                rel = n.get("relationship", "RELATED")
                neighbor = n.get("node", {})
                n_type = neighbor.get("type", "")
                n_name = (
                    neighbor.get("name", "")
                    or neighbor.get("name_en", "")
                    or neighbor.get("title", "")
                    or neighbor.get("title_en", "")
                    or neighbor.get("gate_title", "")
                    or neighbor.get("tier", "")
                    or neighbor.get("term", "")
                    or neighbor.get("question", "")[:60]
                    or str(neighbor.get("id", ""))
                )
                direction = n.get("direction", "")
                arrow = "←" if direction == "incoming" else "→"
                lines.append(f"    {arrow} {rel} {arrow} [{n_type}] {n_name}")

        return "\n".join(lines)


# Try importing VectorRetriever
try:
    from vector_db import VectorRetriever
    HAS_VECTOR_DB = True
except Exception:
    HAS_VECTOR_DB = False


class GraphRAGQA:
    """
    Hybrid Vector DB + Knowledge Graph Question Answering Pipeline.
    Supports bilingual English & Bengali queries with 100% grounded facts.
    """

    # Verified festival calendar anchor
    FESTIVAL_CALENDAR = """
[Official Festival Calendar 2026 / পুজোর নির্ঘণ্ট ২০২৬]
- Mahalaya (মহালয়া): Saturday, 10 October 2026 (04:00 AM Mahishasuramardini, Tarpan, Chokkhu Daan)
- Maha Shashthi (মহাষষ্ঠী): Friday, 16 October 2026 (Bodhon, Astradan, Amantran & Adhibas by 8:30 AM; Inauguration & Sandhya Arati 6:30 PM)
- Maha Saptami (মহাসপ্তমী): Saturday 17 & Sunday 18 October 2026 (17 Oct 7:00 AM Nabapatrika Snan; 18 Oct by 5:53 AM Saptami Puja; Sandhya Arati 6:30 PM)
- Maha Ashtami (মহাষ্টমী): Monday, 19 October 2026 (Pushpanjali at 6:00 AM; Sandhi Puja from 7:26 AM to 8:14 AM; Sandhya Arati 6:30 PM)
- Maha Nabami (মহানবমী): Tuesday, 20 October 2026 (Nabami Puja and Homa at 8:30 AM; Grand Bhog at noon; Sandhya Arati & Cultural Evening 6:30 PM)
- Bijoya Dashami (বিজয়া দশমী): Wednesday, 21 October 2026 (Dashami Puja by 8:31 AM; Debi Boron & Sindoor Khela at 12:00 PM; Bisarjan at 6:00 PM)
"""

    def __init__(self, retriever: GraphRAGRetriever, model: Optional[str] = None):
        self.retriever = retriever
        self.client = None
        self.vector_retriever = None

        openrouter_key = os.environ.get("OPENROUTER_API_KEY")
        openai_key = os.environ.get("OPENAI_API_KEY")

        if openrouter_key and HAS_OPENAI:
            self.client = openai.OpenAI(
                base_url="https://openrouter.ai/api/v1",
                api_key=openrouter_key,
            )
            self.model = model or "nvidia/nemotron-3-ultra-550b-a55b:free"
            print(f"  Using LLM: {self.model} via OpenRouter")
        elif openai_key and HAS_OPENAI:
            self.client = openai.OpenAI(api_key=openai_key)
            self.model = model or "gpt-4o"
            print(f"  Using LLM: {self.model} via OpenAI")
        else:
            self.model = model or "nvidia/nemotron-3-ultra-550b-a55b:free"

        if HAS_VECTOR_DB:
            try:
                self.vector_retriever = VectorRetriever()
            except Exception as e:
                print(f"  Note: ChromaDB vector store not loaded ({e}), using graph retriever.")

    def answer(self, question: str) -> Dict[str, Any]:
        """
        Answer a question using Hybrid Vector DB + Graph RAG:
        1. Retrieve semantic vector chunks from ChromaDB (supports Bengali & English)
        2. Retrieve graph subgraphs and relationships
        3. Ground with verified festival calendar
        4. Generate response in user's language (Bangla or English)
        """
        context_parts = []

        # 1. Vector Search (ChromaDB)
        if self.vector_retriever:
            try:
                vector_results = self.vector_retriever.search(question, top_k=3)
                for vr in vector_results:
                    context_parts.append(vr["text"])
            except Exception:
                pass

        # 2. Graph Retrieval
        graph_context = self.retriever.retrieve_for_query(question)
        if graph_context and graph_context != "No relevant information found in knowledge graph.":
            context_parts.append(graph_context)

        # 3. Calendar anchor if question relates to date, day, ritual, schedule
        q_lower = question.lower()
        date_keywords = [
            "when", "date", "day", "time", "schedule", "mahalaya", "shashthi", "saptami",
            "ashtami", "nabami", "navami", "dashami", "sandhi", "anjali", "tarikh", "kobe",
            "তারিখ", "কবে", "সময়", "নবমী", "অষ্টমী", "সপ্তমী", "ষষ্ঠী", "মহালয়া", "দশমী",
            "সন্ধি", "পুজো", "পূজা", "পুজোর দিন", "কখন"
        ]
        if any(kw in q_lower or kw in question for kw in date_keywords):
            context_parts.insert(0, self.FESTIVAL_CALENDAR)

        combined_context = "\n\n---\n\n".join(context_parts)

        if not self.client:
            return {
                "question": question,
                "answer": "API key not configured. Please ensure OPENROUTER_API_KEY is configured in your .env file.",
                "follow_up_questions": [],
                "context": combined_context,
                "llm_used": False,
            }

        prompt = f"""You are the official AI guide for the IISc Sharodiya Durgotsab (Durga Puja at the Indian Institute of Science, Bengaluru).
You have access to the verified scraped website data, ChromaDB vector store, and Knowledge Graph.

CRITICAL INSTRUCTIONS:
1. OUTPUT FORMAT:
   - You must respond strictly with a single valid JSON object.
   - Do NOT include any markdown codeblocks (no ```json or ```), no asterisks (**), and no commentary outside the JSON.
   - Structure:
{{
  "answer": "concise factual answer string without asterisks",
  "follow_up_questions": [
    "question 1",
    "question 2",
    "question 3",
    "question 4"
  ]
}}

2. CONCISENESS:
   - Be direct, factual, and strictly to the point with the "answer" field.
   - Answer ONLY what was asked. No unrequested filler.

3. RECOMMENDED FOLLOW-UP QUESTIONS:
   - For ANY question asked (including when the user asks a follow-up question), in "follow_up_questions" provide exactly 3 to 4 fresh, natural, helpful follow-up questions directly relevant to what was asked.
   - If responding in Bengali (বাংলা), provide both the answer and the follow-up questions in Bengali.
   - If the user asked in Bengali transliteration (Banglish), answer in clean English or Banglish.

4. ACCURACY & KEY FACTS:
   - Mahalaya: Saturday, 10 October 2026
   - Maha Shashthi: Friday, 16 October 2026
   - Maha Saptami: Saturday 17 & Sunday 18 October 2026
   - Maha Ashtami: Monday, 19 October 2026 (Pushpanjali 6:00 AM, Sandhi Puja 7:26 to 8:14 AM)
   - Maha Nabami: Tuesday, 20 October 2026 (Puja & Homa 8:30 AM)
   - Bijoya Dashami: Wednesday, 21 October 2026 (Debi Boron & Sindoor Khela 12:00 PM, Bisarjan 6:00 PM)
   - Venue: Tata Memorial Club (TMC) Ground, opposite SBI branch, IISc Bengaluru (Map grid F3 126).
   - Convenors: Devraj Karmakar (+91 7384859512), Sirshendu Pathak (+91 7001453229).
   - General Secretaries: Tathagata Ghosh (+91 7890825610), Arnab Ghosh.
   - T-shirt: ₹449 on Meme Theory (Petrol Blue, sizes S-3XL).
   - Donations: UPI to iiscsdc@sbi.

5. STRICT TOPIC GUARDRAIL:
   - Your ONLY scope is the IISc Sharodiya Durgotsab (Durga Puja at IISc Bengaluru).
   - If the user asks about ANYTHING unrelated to the festival (e.g. computer science, coding, binary trees, general math, movies, politics, etc.), you must respond strictly with:
{{
  "answer": "I am not sure about that, please contact the IISc Sharodiya Durgotsab Committee.",
  "follow_up_questions": []
}}
     (Or in Bengali if asked in Bengali:
{{
  "answer": "আমি এ বিষয়ে নিশ্চিত নই, দয়া করে আইআইএসসি শারদীয়া দুর্গোৎসব কমিটির সাথে যোগাযোগ করুন।",
  "follow_up_questions": []
}})

--- RETRIEVED CONTEXT ---
{combined_context}
------------------------

USER QUESTION: {question}

JSON RESPONSE:"""

        system_message = (
            "You are the official assistant for IISc Sharodiya Durgotsab. "
            "You must respond strictly with a valid JSON object containing 'answer' and 'follow_up_questions' (list of 3-4 strings). "
            "Always suggest 3-4 fresh follow-up questions, even if the question being asked is itself a follow-up. "
            "Never use asterisks (**) for bolding. "
            "Strictly answer only questions about IISc Sharodiya Durgotsab in English, Bengali, or Bengali transliteration (Banglish). "
            "For any off-topic question, return JSON with 'I am not sure about that, please contact the IISc Sharodiya Durgotsab Committee.' and empty follow_up_questions []."
        )

        # Fallback chain: primary requested model first, then lightning if free tier is busy/limits
        candidate_models = [self.model]
        if self.model != "nvidia/nemotron-3.5-lightning":
            candidate_models.append("nvidia/nemotron-3.5-lightning")

        for m in candidate_models:
            try:
                resp = self.client.chat.completions.create(
                    model=m,
                    messages=[
                        {"role": "system", "content": system_message},
                        {"role": "user", "content": prompt},
                    ],
                    temperature=0.0,
                )
                if resp and resp.choices and len(resp.choices) > 0:
                    msg = resp.choices[0].message
                    content = msg.content
                    if not content and hasattr(msg, "reasoning") and msg.reasoning:
                        content = msg.reasoning
                    if content and content.strip():
                        clean_answer, follow_ups = self._parse_answer_and_follow_ups(content)
                        return {
                            "answer": clean_answer,
                            "follow_up_questions": follow_ups,
                            "question": question,
                            "context": combined_context,
                            "llm_used": True,
                            "model_used": m,
                        }
            except Exception as e:
                # Log to debug but do not print raw data dump
                continue

        # If all candidates fail, return clean graceful message (NEVER dump raw database text)
        return {
            "answer": "I am not sure about that, please contact the IISc Sharodiya Durgotsab Committee.",
            "follow_up_questions": [],
            "question": question,
            "context": combined_context,
            "llm_used": False,
        }

    def answer_json(self, question: str) -> str:
        """Return formatted JSON string with answer and follow_up_questions."""
        res = self.answer(question)
        payload = {
            "answer": res["answer"],
            "follow_up_questions": res.get("follow_up_questions", []),
        }
        return json.dumps(payload, indent=2, ensure_ascii=False)

    @staticmethod
    def _parse_answer_and_follow_ups(raw_text: str) -> tuple[str, list[str]]:
        """
        Parses the model output into a clean answer string and a list of follow-up questions.
        Supports direct JSON parsing and marker-based fallback.
        """
        cleaned_text = raw_text.replace("**", "").replace("__", "").strip()

        # 1. Try JSON parsing (strip ```json ... ``` code fence if present)
        json_candidate = cleaned_text
        if json_candidate.startswith("```"):
            json_candidate = json_candidate.strip("`")
            if json_candidate.startswith("json"):
                json_candidate = json_candidate[4:].strip()

        try:
            start_idx = json_candidate.find("{")
            end_idx = json_candidate.rfind("}")
            if start_idx != -1 and end_idx != -1 and end_idx > start_idx:
                data = json.loads(json_candidate[start_idx : end_idx + 1])
                if "answer" in data:
                    ans = str(data["answer"]).replace("**", "").replace("__", "").strip()
                    fu = [
                        str(q).replace("**", "").replace("__", "").strip()
                        for q in data.get("follow_up_questions", [])
                        if q
                    ]
                    return ans, fu[:4]
        except Exception:
            pass

        # 2. Fallback marker-based extraction
        answer_text = cleaned_text
        follow_ups: list[str] = []

        markers = [
            "FOLLOW_UP_QUESTIONS:",
            "FOLLOW UP QUESTIONS:",
            "Suggested follow-up questions:",
            "Suggested follow up questions:",
            "প্রাসঙ্গিক পরবর্তী প্রশ্ন:",
            "প্রাসঙ্গিক প্রশ্ন:",
        ]

        found_marker = None
        for m in markers:
            if m in cleaned_text:
                found_marker = m
                break

        if found_marker:
            parts = cleaned_text.split(found_marker, 1)
            answer_part = parts[0].strip()
            if answer_part.startswith("ANSWER:"):
                answer_part = answer_part[len("ANSWER:"):].strip()
            answer_text = answer_part

            follow_up_text = parts[1].strip()
            for line in follow_up_text.split("\n"):
                line = line.strip()
                if not line:
                    continue
                cleaned_line = re.sub(r"^(\d+[\.\)]|[-*•]|\[\d+\])\s*", "", line).strip()
                if cleaned_line and len(cleaned_line) > 4 and not any(
                    cleaned_line.lower().startswith(x) for x in ["answer:", "follow_up", "note:"]
                ):
                    follow_ups.append(cleaned_line)
        else:
            if answer_text.startswith("ANSWER:"):
                answer_text = answer_text[len("ANSWER:"):].strip()

        return answer_text.strip(), follow_ups[:4]


def demo_queries(qa_system: GraphRAGQA):
    """Run demonstration queries covering key aspects of the website."""
    sample_queries = [
        "When is the Durga Puja at IISc in 2026, and what are the main dates?",
        "Where is the pandal located, and how do I reach it from Yeshwantpur or the Main Gate?",
        "What happens on Maha Ashtami and when is Sandhi Puja?",
        "Is there an official T-shirt? How much is it and how do I buy it?",
        "Who is organizing it and who are the convenors I can contact?",
        "How can I volunteer for the festival and what teams are there?",
        "What are the sponsorship tiers and what does Genotypic Group sponsor?",
        "How do donations work and how do I get a receipt?",
        "What is the Probash magazine and how can I submit an article?",
        "What ambient soundscapes and music are featured on the website?",
    ]

    for q in sample_queries:
        print(f"\n{'='*75}")
        print(f"QUESTION: {q}")
        print(f"{'='*75}")
        result = qa_system.answer(q)
        payload = {
            "answer": result["answer"],
            "follow_up_questions": result.get("follow_up_questions", []),
        }
        print(json.dumps(payload, indent=2, ensure_ascii=False))


def main():
    import sys

    project_dir = os.path.dirname(os.path.abspath(__file__))
    graph_json = os.path.join(project_dir, "graph.json")

    neo4j_uri = os.environ.get("NEO4J_URI")
    neo4j_user = os.environ.get("NEO4J_USER", "neo4j")
    neo4j_password = os.environ.get("NEO4J_PASSWORD")

    if neo4j_uri and HAS_NEO4J:
        retriever = GraphRAGRetriever(
            backend="neo4j",
            neo4j_uri=neo4j_uri,
            neo4j_user=neo4j_user,
            neo4j_password=neo4j_password,
        )
    elif os.path.exists(graph_json):
        retriever = GraphRAGRetriever(backend="networkx", graph_path=graph_json)
    else:
        print("Graph file not found. Run scraper.py then graph_builder.py first.")
        sys.exit(1)

    qa = GraphRAGQA(retriever)

    # Check for direct interactive flag
    if "--interactive" not in sys.argv and "-i" not in sys.argv and "--interactive-only" not in sys.argv:
        demo_queries(qa)

    print("\n" + "=" * 75)
    print("IISc Sharodiya Durgotsab QA Assistant (Interactive JSON Mode)")
    print("Type your question below (or type 'quit' to exit):")
    print("=" * 75)

    while True:
        try:
            query = input("\nYour Question: ").strip()
        except (EOFError, KeyboardInterrupt):
            break

        if not query or query.lower() in ("quit", "exit", "q"):
            break

        # Pass any question directly (initial question or follow-up question)
        res = qa.answer(query)
        payload = {
            "answer": res["answer"],
            "follow_up_questions": res.get("follow_up_questions", []),
        }
        print(f"\n{json.dumps(payload, indent=2, ensure_ascii=False)}\n")

    retriever.close()


if __name__ == "__main__":
    main()
