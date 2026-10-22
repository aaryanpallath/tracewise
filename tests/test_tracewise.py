import json

import pytest

from tracewise import cli, mcp_server, trace
from tracewise import metrics as M
from tracewise.corpus import load, parse_java, split
from tracewise.llm import Reply
from tracewise.search import Index, tokens
from tracewise.store import ReviewError, Store

ACTION = """package edu.ncsu.csc.itrust.action;

import edu.ncsu.csc.itrust.dao.mysql.PatientDAO;

/**
 * Handles editing a patient's demographics. Used by editPatient.jsp.
 */
public class EditPatientAction {
\tprivate PatientDAO patientDAO;

\t/**
\t * Updates the patient's demographic information. Sends an email.
\t * @param p the patient
\t */
\tpublic void updateInformation(PatientBean p) throws iTrustException {
\t\tpatientDAO.editPatient(p);
\t}

\tprivate String formatName(String first, String last) {
\t\treturn first + " " + last;
\t}
}
"""
DAO = """package edu.ncsu.csc.itrust.dao.mysql;
/** Used for managing patient demographics in the patients table. */
public class PatientDAO {
\t/** Edits a patient's demographic record. */
\tpublic void editPatient(PatientBean p) throws DBException {
\t}
}
"""
LOG = """package edu.ncsu.csc.itrust.dao.mysql;
/** Logs transactions for the access log. */
public class TransactionDAO {
\tpublic List<TransactionBean> getAllTransactions() throws DBException { return null; }
}
"""


@pytest.fixture
def dataset(tmp_path):
    d = tmp_path / "itrust"
    (d / "req").mkdir(parents=True)
    (d / "code" / "action").mkdir(parents=True)
    (d / "code" / "dao" / "mysql").mkdir(parents=True)
    (d / "req" / "UC4S1.txt").write_text("A patient may edit their own demographic information.\n")
    (d / "req" / "UC4E1.txt").write_text("The patient chosen is not the desired patient.\n")
    (d / "req" / "UC8S1.txt").write_text("A patient views the access log of transactions.\n")
    (d / "code" / "action" / "EditPatientAction.java").write_text(ACTION)
    (d / "code" / "dao" / "mysql" / "PatientDAO.java").write_text(DAO)
    (d / "code" / "dao" / "mysql" / "TransactionDAO.java").write_text(LOG)
    (d / "itrust_solution_links.txt").write_text(
        "UC4S1.txt: EditPatientAction.java\nUC4S1.txt: PatientDAO.java\nUC8S1.txt: TransactionDAO.java\n"
    )
    (d / "itrust_raw_callgraph.txt").write_text(
        "C:edu.ncsu.csc.itrust.action.EditPatientAction edu.ncsu.csc.itrust.dao.mysql.PatientDAO\n"
        "C:edu.ncsu.csc.itrust.action.EditPatientAction java.lang.String\n"
    )
    return load(d)


def test_load_parses_requirements_code_gold_and_call_graph(dataset):
    assert set(dataset.requirements) == {"UC4S1", "UC4E1", "UC8S1"}
    assert dataset.gold["UC4S1"] == {"EditPatientAction", "PatientDAO"} and dataset.gold["UC4E1"] == set()
    c = dataset.classes["EditPatientAction"]
    assert c.package == "action" and c.doc == "Handles editing a patient's demographics. Used by editPatient.jsp."
    assert [m.signature for m in c.methods] == [
        "public void updateInformation(PatientBean p)",
        "private String formatName(String first, String last)",
    ]
    assert c.methods[0].doc == "Updates the patient's demographic information."
    assert dataset.related("PatientDAO") == {"uses": [], "used_by": ["EditPatientAction"]}
    assert [r.id for r in dataset.flows("UC4")] == ["UC4E1", "UC4S1"]


def test_parse_java_without_comments():
    c = parse_java("X", "", "public class X {\n  public int size() { return 0; }\n}\n")
    assert c.doc == "" and [m.signature for m in c.methods] == ["public int size()"]


def test_split_is_by_use_case():
    assert split("UC4S1") == split("UC40E2") == "dev" and split("UC10S1") == split("UC41") == "test"


def test_tokens_split_identifiers_and_stem():
    assert tokens("getPatientDAO editsOfficeVisits", code=True) == ["patient", "dao", "edit", "office", "visit"]
    assert tokens("HCPAction") == ["hcp", "action"]


def test_vsm_ranks_by_shared_vocabulary(dataset):
    index = Index(dataset, use_embeddings=False)
    assert index.search("transactions access log", 1, "hybrid") == ["TransactionDAO"]  # falls back to vsm
    assert index.search("edit patient demographic", 2, "vsm")[0] in {"EditPatientAction", "PatientDAO"}


def test_set_metrics_and_ranking_metrics():
    gold = {"A": {"x", "y"}, "B": {"z"}, "C": set()}
    pred = {"A": {"x", "q"}, "B": set(), "C": set()}
    m = M.set_metrics(pred, gold)
    assert (m["precision"], m["recall"], m["proposed"], m["correct"]) == (0.5, 0.333, 2, 1)
    assert m["empty_kept_empty"] == "1/1"
    assert M.average_precision(["x", "q", "y"], {"x", "y"}) == pytest.approx((1 + 2 / 3) / 2)
    assert M.recall_at({"A": ["x", "q", "y"], "B": ["z"]}, {"A": {"x", "y"}, "B": {"z"}}, 2) == pytest.approx(0.667)


def test_link_parsing_tolerates_key_names_and_flags_invented_classes(dataset):
    links, invalid = trace._parse_links(
        '```json\n{"links": [{"action": "EditPatientAction", "reason": "edits"}, {"class": "PatientDAO.java"},'
        ' {"class": "MadeUpDAO", "reason": "?"}]}\n```',
        dataset,
    )
    assert links == {"EditPatientAction": "edits", "PatientDAO": ""} and invalid == ["MadeUpDAO"]
    assert trace._parse_links("no json here", dataset) == ({}, [])


def scripted(replies):
    calls = []

    def fake_chat(messages, tools=None, **kw):
        calls.append(list(messages))  # the agent extends one list; keep what each call saw
        return replies[len(calls) - 1]

    return fake_chat, calls


def call(tool, **args):
    return {"function": {"name": tool, "arguments": args}}


def test_agent_uses_tools_then_submits(dataset, monkeypatch):
    index = Index(dataset, use_embeddings=False)
    fake, calls = scripted(
        [
            Reply("", [call("related_classes", name="EditPatientAction")]),
            Reply(
                "",
                [
                    call(
                        "submit_links",
                        links=[
                            {"class": "EditPatientAction", "reason": "updates demographics"},
                            {"class": "PatientDAO", "reason": "used by the action to save them"},
                        ],
                    )
                ],
            ),
        ]
    )
    monkeypatch.setattr(trace, "chat", fake)
    p = trace.agent(dataset, index, "UC4S1")
    assert set(p.links) == {"EditPatientAction", "PatientDAO"} and p.steps == 2 and not p.error
    # the conversation starts with a search already done, and the tool result reached the model
    assert calls[0][3]["role"] == "tool" and "EditPatientAction" in calls[0][3]["content"]
    assert "PatientDAO (dao.mysql)" in calls[1][-1]["content"]
    # requirement context includes the other flows of the use case
    assert "UC4E1" in calls[0][1]["content"]


def test_agent_step_limit_falls_back_to_json(dataset, monkeypatch):
    index = Index(dataset, use_embeddings=False)
    loop = [Reply("", [call("read_class", name="Nope")]) for _ in range(2)]
    final = Reply('{"links": [{"name": "TransactionDAO", "reason": "logs"}]}')
    fake, calls = scripted([*loop, final])
    monkeypatch.setattr(trace, "chat", fake)
    p = trace.agent(dataset, index, "UC8S1", max_steps=2)
    assert p.links == {"TransactionDAO": "logs"} and p.error == "step limit"
    assert "no class named 'Nope'" in calls[1][-1]["content"]


def test_review_queue(tmp_path):
    s = Store(tmp_path / "r.sqlite")
    a, created = s.propose("UC4S1", "PatientDAO", "saves demographics", "agent")
    assert created and s.propose("UC4S1", "PatientDAO", "again", "agent") == (a, False)
    b, _ = s.propose("UC4S1", "PatientBean", "data holder", "agent")
    s.review(a, "accepted", "reviewer", "checked editPatient")
    s.review(b, "rejected", "reviewer", "bean")
    with pytest.raises(ReviewError, match="already accepted"):
        s.review(a, "rejected", "someone else")
    with pytest.raises(ReviewError, match="reviewer name"):
        s.review(b, "accepted", " ")
    with pytest.raises(ReviewError, match="no link"):
        s.review(999, "accepted", "reviewer")
    assert s.propose("UC4S1", "PatientBean", "again", "agent")[1] is False  # a rejected link stays rejected
    assert s.matrix() == {"UC4S1": ["PatientDAO"]}
    assert s.gaps(["UC4S1", "UC4E1"]) == ["UC4E1"]


def test_cli_import_review_and_matrix(tmp_path, monkeypatch, capsys):
    monkeypatch.setattr(cli, "DB", tmp_path / "r.sqlite")
    run = tmp_path / "run.jsonl"
    run.write_text(json.dumps({"id": "UC4S1", "links": {"PatientDAO": "saves it", "PatientBean": "holds it"}}) + "\n")
    assert cli.main(["import", str(run)]) == 0
    assert cli.main(["accept", "1", "--by", "reviewer"]) == 0
    assert cli.main(["accept", "1", "--by", "reviewer"]) == 1
    assert cli.main(["matrix", "--out", str(tmp_path / "m.csv")]) == 0
    rows = (tmp_path / "m.csv").read_text().splitlines()
    assert len(rows) == 2 and rows[1].startswith("UC4S1,PatientDAO,saves it,agent (run.jsonl),reviewer")


def test_mcp_tools_validate_and_only_propose(dataset, tmp_path, monkeypatch):
    store = Store(tmp_path / "r.sqlite")
    monkeypatch.setattr(mcp_server, "state", lambda: (dataset, Index(dataset, use_embeddings=False), store))
    assert "error" in mcp_server.propose_link("UC99", "PatientDAO", "x")
    assert "error" in mcp_server.propose_link("UC4S1", "Invented", "x")
    assert "error" in mcp_server.propose_link("UC4S1", "PatientDAO", "  ")
    assert mcp_server.propose_link("UC4S1", "PatientDAO", "saves demographics")["status"] == "proposed"
    assert mcp_server.trace_matrix() == {}  # nothing enters the matrix until a person accepts it
    assert set(mcp_server.coverage_gaps()) == {"UC4S1", "UC4E1", "UC8S1"}
    assert "PatientDAO" in mcp_server.search_code("patient demographics")


def test_untrusted_reasons_cannot_inject_escapes_or_formulas(tmp_path, monkeypatch):
    s = Store(tmp_path / "r.sqlite")
    link, _ = s.propose("UC4S1", "PatientDAO", "ok\x1b[2K\rACCEPTED by admin", "agent")
    assert s.get(link)["reason"] == "ok [2K ACCEPTED by admin"  # no escape or carriage return survives
    s.propose("UC4S1", "EditPatientAction", '=HYPERLINK("http://x","click")', "agent")
    for i in (1, 2):
        s.review(i, "accepted", "reviewer")
    monkeypatch.setattr(cli, "DB", tmp_path / "r.sqlite")
    assert cli.main(["matrix", "--out", str(tmp_path / "m.csv")]) == 0
    assert "'=HYPERLINK" in (tmp_path / "m.csv").read_text()
