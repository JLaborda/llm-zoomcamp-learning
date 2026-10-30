import sqlite3
from typing import Any, Dict, List


class SQLiteFTS:

    def __init__(self, db_path: str = ":memory:"):
        """Inicializa la base de datos en memoria o disco."""
        self.conn = sqlite3.connect(db_path)
        self._create_table()

    def _create_table(self):
        """Crea la tabla virtual FTS5 optimizada para BM25."""
        with self.conn:
            self.conn.execute("""
                CREATE VIRTUAL TABLE IF NOT EXISTS faq_docs USING fts5(
                    question,
                    answer,
                    section,
                    course UNINDEXED
                );
            """)

    def index_documents(self, documents: List[Dict[str, Any]]):
        """Indexa la lista de documentos obtenida del JSON del curso."""
        with self.conn:
            self.conn.executemany(
                """
                INSERT INTO faq_docs (question, answer, section, course)
                VALUES (:question, :answer, :section, :course)
            """,
                documents,
            )

    def search(
        self, query: str, course: str | None = None, num_results: int = 5
    ) -> List[Dict[str, Any]]:
        """Realiza búsqueda de texto completo clasificada por ranking BM25."""
        # Limpiamos palabras para evitar errores de sintaxis FTS5
        words = [f'"{word}"' for word in query.replace('"', "").split()]
        fts_query = " OR ".join(words)

        sql = """
            SELECT question, answer, section, course, bm25(faq_docs) AS score
            FROM faq_docs
            WHERE faq_docs MATCH ?
        """
        params: list[Any] = [fts_query]

        if course:
            sql += " AND course = ?"
            params.append(course)

        sql += " ORDER BY rank LIMIT ?"
        params.append(num_results)

        cursor = self.conn.cursor()
        cursor.execute(sql, params)

        results = []
        for row in cursor.fetchall():
            results.append({
                "question": row[0],
                "answer": row[1],
                "section": row[2],
                "course": row[3],
                "score": row[4],
            })
        return results

def close(self):
    """Cierra explícitamente la conexión a la base de datos."""
    self.conn.close()