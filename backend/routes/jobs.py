import logging
import uuid
from urllib.parse import quote

from fastapi import APIRouter, BackgroundTasks, HTTPException, Request
from pydantic import BaseModel

from backend.auth.jwt_handler import decode_token
from backend.utils.api_contracts import build_account_plan_contract_from_marketing_plan
from backend.utils.job_status import create_job, get_job, update_job
from backend.utils.strategy_builder.comprehensive_plan_generator import (
    ACCOUNT_PLAN_PHASE_MESSAGES,
    ACCOUNT_PLAN_PHASE_PROGRESS,
    build_product_marketing_plan,
)

LOGGER = logging.getLogger(__name__)
router = APIRouter()


def _decode_request_token(request: Request) -> None:
    auth_header = request.headers.get("authorization") or request.headers.get("Authorization")
    if not auth_header or " " not in auth_header:
        raise HTTPException(status_code=401, detail="Missing Authorization header")
    token = auth_header.split(" ", 1)[1]
    decoded = decode_token(token)
    if not decoded or not decoded.get("company_id"):
        raise HTTPException(status_code=401, detail="Invalid token")


class AccountPlanJobRequest(BaseModel):
    account_id: str
    product_id: str


@router.post("/jobs/account-plan", status_code=201)
def start_account_plan_job(
    body: AccountPlanJobRequest,
    request: Request,
    background_tasks: BackgroundTasks,
) -> dict[str, str]:
    _decode_request_token(request)
    job_id = str(uuid.uuid4())
    create_job(job_id, job_type="account_plan")
    background_tasks.add_task(
        _run_account_plan_job,
        job_id,
        body.account_id,
        body.product_id,
    )
    return {"job_id": job_id}


@router.get("/jobs/{job_id}")
def job_status(job_id: str, request: Request) -> dict:
    _decode_request_token(request)
    job = get_job(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="Job not found")
    return job.dict()


def _run_account_plan_job(job_id: str, account_id: str, product_id: str) -> None:
    def report_phase(phase: str, message: str, progress: float) -> None:
        try:
            update_job(
                job_id,
                status="running",
                phase=phase,
                message=message,
                progress=progress,
            )
        except KeyError:
            LOGGER.warning("Job %s missing while reporting phase %s", job_id, phase)

    try:
        initial_message = ACCOUNT_PLAN_PHASE_MESSAGES.get("account_context", "Loading account history and engagement signals…")
        initial_progress = ACCOUNT_PLAN_PHASE_PROGRESS.get("account_context", 5.0)
        update_job(
            job_id,
            status="running",
            phase="account_context",
            message=initial_message,
            progress=initial_progress,
        )
        plan = build_product_marketing_plan(
            product_id,
            account_id=account_id,
            job_status_callback=report_phase,
        )
        accounts = plan.get("accounts") or []
        account_entry = next(
            (
                acct
                for acct in accounts
                if acct.get("account_id") == account_id and not acct.get("error")
            ),
            None,
        )
        if not account_entry and accounts:
            account_entry = accounts[0]
        if not account_entry:
            raise ValueError(f"Account plan missing for {account_id}")
        build_account_plan_contract_from_marketing_plan(
            plan,
            account_entry,
            product_id=product_id,
        )
        result_location = f"/accounts/{account_id}/plan?product_id={quote(product_id)}"
        done_message = ACCOUNT_PLAN_PHASE_MESSAGES.get("done", "Account plan ready with clear next steps.")
        update_job(
            job_id,
            status="completed",
            phase="done",
            message=done_message,
            progress=100.0,
            result_location=result_location,
        )
    except Exception as exc:
        LOGGER.exception("Account plan job %s failed", job_id, exc_info=exc)
        failure_message = (
            "We failed while building the account plan. Click 'Show error' for details."
        )
        try:
            update_job(
                job_id,
                status="failed",
                phase="done",
                message=failure_message,
                progress=100.0,
                error=str(exc),
            )
        except KeyError:
            LOGGER.warning("Unable to update failed status for job %s", job_id)
