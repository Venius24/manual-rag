import db
import keyword_search as ks


class RecordingCursor:
    def __init__(self, log, rows=()):
        self.log, self.rows = log, rows

    def __enter__(self):
        return self

    def __exit__(self, *a):
        return False

    def execute(self, sql, params=None):
        self.log.append((" ".join(sql.split()), params))

    def fetchall(self):
        return list(self.rows)


class RecordingConn(RecordingCursor):
    def __init__(self, log, rows=()):
        super().__init__(log, rows)
        self.commits = 0

    def cursor(self):
        return RecordingCursor(self.log, self.rows)

    def commit(self):
        self.commits += 1


def test_init_schema_is_idempotent_ddl_with_fts_index():
    log = []
    conn = RecordingConn(log)
    db.init_schema(conn)
    statements = [sql for sql, _ in log]
    assert all("IF NOT EXISTS" in s for s in statements)
    assert any("GENERATED ALWAYS AS (to_tsvector('english', text)) STORED" in s for s in statements)
    assert any("USING GIN (tsv)" in s for s in statements)
    assert conn.commits == 1


def test_keyword_search_uses_stored_tsvector(monkeypatch):
    log = []
    rows = [(3, "text", 0.5)]
    monkeypatch.setattr(ks.psycopg, "connect", lambda url: RecordingConn(log, rows))
    assert ks.keyword_search("limits", limit=7) == rows
    sql, params = log[0]
    assert "tsv @@" in sql and "to_tsvector" not in sql
    assert params == ("limits", "limits", 7)
