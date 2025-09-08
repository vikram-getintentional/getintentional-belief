import os
import json
import pytest
from fastapi.testclient import TestClient

from backend.main import app
from backend.auth.jwt_handler import create_access_token


def auth_headers(company_id: str = "testco"):
    token = create_access_token({"company_id": company_id})
    return {"Authorization": f"Bearer {token}"}


@pytest.fixture(scope="module")
def client():
    return TestClient(app)


def test_graph_editor_e2e(client: TestClient):
    # Use seeded test product graph
    product_lookup_id = "test-product-1"

    # 1) Create a capability
    resp = client.post(
        "/graph/capability",
        headers=auth_headers(),
        json={"product_id": product_lookup_id, "name": "User Management", "description": "Create & manage users"},
    )
    assert resp.status_code == 200, resp.text
    cap = resp.json()
    cap_id = cap["id"]
    assert cap_id

    # 2) Update capability attributes (name, description, coreness, buying_likelihood)
    resp = client.put(
        f"/graph/capability/{cap_id}",
        headers=auth_headers(),
        json={
            "product_id": product_lookup_id,
            "name": "User & Access Management",
            "description": "CRUD users and roles",
            "coreness": "Core",
            "buying_likelihood": 75,
        },
    )
    assert resp.status_code == 200, resp.text

    # 3) List capabilities and confirm presence
    resp = client.get(
        f"/graph/nodes/capability?product_id={product_lookup_id}",
        headers=auth_headers(),
    )
    assert resp.status_code == 200
    nodes = resp.json().get("nodes", [])
    assert any(n["id"] == cap_id for n in nodes)
    # Buying likelihood reflected via node or edge readout
    node = next(n for n in nodes if n["id"] == cap_id)
    assert (node.get("buying_likelihood") is not None) and (float(node.get("buying_likelihood")) >= 0)

    # 3b) Create a second capability to exercise merge
    resp2 = client.post(
        "/graph/capability",
        headers=auth_headers(),
        json={"product_id": product_lookup_id, "name": "User Provisioning", "description": "Provision users"},
    )
    assert resp2.status_code == 200, resp2.text
    cap2_id = resp2.json()["id"]
    assert cap2_id and cap2_id != cap_id

    # 4) Create a pain family for this capability with new pain/job/persona
    resp = client.post(
        "/graph/pain_family",
        headers=auth_headers(),
        json={
            "product_id": product_lookup_id,
            "capability_id": cap_id,
            "pain": "Slow onboarding",
            "job": "Streamline onboarding",
            "persona": {"title": "Admin", "department": "Operations", "seniority": "Senior", "linkedin_url": "https://www.linkedin.com/in/example"},
            "relevance": "Critical",
            "likelihood": 80,
        },
    )
    assert resp.status_code == 200, resp.text
    pf = resp.json()
    pain_id, job_id, persona_id = pf["pain_id"], pf["job_id"], pf["persona_id"]
    assert pain_id and job_id and persona_id

    # 5) List pain families and verify
    resp = client.get(
        f"/graph/pain_families?product_id={product_lookup_id}&capability_id={cap_id}",
        headers=auth_headers(),
    )
    assert resp.status_code == 200
    families = resp.json().get("items", [])
    assert any(item.get("pain", {}).get("id") == pain_id for item in families)

    # 6) Update family edge attributes
    resp = client.put(
        f"/graph/pain_family/any",
        headers=auth_headers(),
        json={
            "product_id": product_lookup_id,
            "capability_id": cap_id,
            "pain_id": pain_id,
            "job_id": job_id,
            "persona_id": persona_id,
            "relevance": "Supportive",
            "likelihood": 65,
        },
    )
    assert resp.status_code == 200, resp.text

    # 7) Update persona anchor
    resp = client.put(
        f"/graph/persona/{persona_id}",
        headers=auth_headers(),
        json={"product_id": product_lookup_id, "linkedin_url": "https://www.linkedin.com/in/test-user"},
    )
    assert resp.status_code == 200, resp.text

    # 8) Update job anchor
    resp = client.put(
        f"/graph/job/{job_id}",
        headers=auth_headers(),
        json={"product_id": product_lookup_id, "linkedin_url": "https://www.linkedin.com/in/test-job"},
    )
    assert resp.status_code == 200, resp.text

    # 9) Confirm persona appears with linkedin_url in node listing
    resp = client.get(
        f"/graph/nodes/persona?product_id={product_lookup_id}", headers=auth_headers()
    )
    assert resp.status_code == 200
    people = resp.json().get("nodes", [])
    assert any(n["id"] == persona_id and n.get("linkedin_url") for n in people)

    # 10) Merge cap2 into cap1 and then delete cap2 (should already be removed by merge)
    resp = client.post(
        "/graph/capability/merge",
        headers=auth_headers(),
        json={"product_id": product_lookup_id, "source_id": cap2_id, "target_id": cap_id},
    )
    assert resp.status_code == 200, resp.text

    # Delete (idempotent check; if node already gone, expect 404)
    resp = client.delete(
        f"/graph/capability/{cap2_id}?product_id={product_lookup_id}",
        headers=auth_headers(),
    )
    # allow either 200 or 404 depending on merge removal timing
    assert resp.status_code in (200, 404)
