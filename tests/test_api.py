from fastapi.testclient import TestClient

from liveindex.api.app import create_app


def test_api_epochs_and_search(log, emb, monkeypatch):
    monkeypatch.setenv("ALLOW_TEST_PRINCIPAL", "1")
    h = {"X-Test-Principal": "u"}
    with TestClient(create_app(emb, log)) as c:
        e1 = c.post("/v1/docs", json={"doc_id": "a", "text": "apple pie recipe", "allowed": ["u"]}).json()["epoch"]
        c.post("/v1/docs", json={"doc_id": "b", "text": "banana bread", "allowed": ["u"]})
        r = c.post("/v1/search", json={"query": "apple", "k": 2, "min_epoch": e1}, headers=h).json()
        assert r["hits"][0]["doc_id"] == "a" and r["epoch"] >= e1
        ep = c.get("/v1/epoch").json()
        assert ep["committed"] == 2
        e3 = c.delete("/v1/docs/a").json()["epoch"]
        r = c.post("/v1/search", json={"query": "apple", "k": 2, "min_epoch": e3}, headers=h).json()
        assert all(x["doc_id"] != "a" for x in r["hits"])
        assert c.post("/v1/search", json={"query": "apple"}).status_code == 401
