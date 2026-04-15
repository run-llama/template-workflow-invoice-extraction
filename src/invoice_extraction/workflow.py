import os
from pathlib import Path
from typing import Annotated

from llama_cloud import AsyncLlamaCloud
from llama_cloud.types.extraction.extract_config_param import ExtractConfigParam
from pydantic import BaseModel, Field
from workflows import Context, Workflow, step
from workflows.events import (
    HumanResponseEvent,
    InputRequiredEvent,
    StartEvent,
    StopEvent,
)
from workflows.resource import Resource


class InvoiceData(BaseModel):
    invoice_date: str = Field(description="Date on the invoice")
    customer: str = Field(description="Customer reported on the invoice")
    amount_due: float = Field(description="Amount due")


class FeedbackRequiredEvent(InputRequiredEvent):
    extraction_result: str


class HumanFeedbackEvent(HumanResponseEvent):
    approved: bool


# required for all llama cloud calls
LLAMA_CLOUD_API_KEY = os.getenv("LLAMA_CLOUD_API_KEY")
# get this in case running against a different environment than production
LLAMA_CLOUD_BASE_URL = os.getenv("LLAMA_CLOUD_BASE_URL")
LLAMA_CLOUD_PROJECT_ID = os.getenv("LLAMA_DEPLOY_PROJECT_ID")


async def get_llama_cloud_client(*args, **kwargs) -> AsyncLlamaCloud:
    return AsyncLlamaCloud(
        api_key=LLAMA_CLOUD_API_KEY,
        base_url=LLAMA_CLOUD_BASE_URL,
        default_headers={"Project-Id": LLAMA_CLOUD_PROJECT_ID}
        if LLAMA_CLOUD_PROJECT_ID
        else {},
    )


class InvoiceExtractWorkflow(Workflow):
    @step
    async def invoice_extraction(
        self,
        ev: StartEvent,
        ctx: Context,
        client: Annotated[AsyncLlamaCloud, Resource(get_llama_cloud_client)],
    ) -> FeedbackRequiredEvent:
        async with ctx.store.edit_state() as state:
            state.extraction_mode = ev.extraction_mode
            state.path = ev.path

        config: ExtractConfigParam
        if ev.extraction_mode == "base":
            config = {
                "extraction_mode": "FAST",
                "high_resolution_mode": False,
                "invalidate_cache": False,
                "cite_sources": False,
                "use_reasoning": False,
                "confidence_scores": False,
            }
        elif ev.extraction_mode == "advanced":
            config = {
                "extraction_mode": "MULTIMODAL",
                "high_resolution_mode": True,
                "invalidate_cache": False,
                "cite_sources": False,
                "use_reasoning": True,
                "confidence_scores": False,
            }
        else:
            config = {
                "extraction_mode": "PREMIUM",
                "high_resolution_mode": True,
                "invalidate_cache": False,
                "cite_sources": True,
                "use_reasoning": True,
                "confidence_scores": True,
            }

        uploaded = await client.files.create(
            file=Path(ev.path).open("rb"),
            purpose="extract",
        )
        result = await client.extraction.extract(
            config=config,
            data_schema=InvoiceData.model_json_schema(),
            file_id=uploaded.id,
        )
        extracted_data: list[InvoiceData] = []
        if isinstance(result.data, list):
            for r in result.data:
                extracted_data.append(InvoiceData.model_validate(r))
        elif result.data is not None:
            extracted_data.append(InvoiceData.model_validate(result.data))
        extraction_result = "\\n\\n---\\n\\n".join(
            [
                f"Invoice Date: {d.invoice_date}\\nCustomer: {d.customer}\\nAmount Due: {d.amount_due}"
                for d in extracted_data
            ]
        )
        async with ctx.store.edit_state() as state:
            state.extraction_result = extraction_result
        return FeedbackRequiredEvent(extraction_result=extraction_result)

    @step
    async def human_feedback(
        self, ev: HumanFeedbackEvent, ctx: Context
    ) -> StopEvent | StartEvent:
        state = await ctx.store.get_state()
        if ev.approved:
            return StopEvent(result=state.extraction_result)
        else:
            return StartEvent(path=state.path, extraction_mode=state.extraction_mode)  # type: ignore


async def main(path: str, extraction_mode: str) -> None:
    w = InvoiceExtractWorkflow(timeout=1800, verbose=False)
    handler = w.run(path=path, extraction_mode=extraction_mode)
    async for ev in handler.stream_events():
        if isinstance(ev, FeedbackRequiredEvent):
            print("Extraction Result:\\n\\n" + ev.extraction_result + "\\n\\n")
            res = input("Approve? [yes/no]: ")
            if res.lower().strip() == "yes":
                handler.ctx.send_event(HumanFeedbackEvent(approved=True))
            else:
                handler.ctx.send_event(HumanFeedbackEvent(approved=False))
    result = await handler
    print(str(result))


workflow = InvoiceExtractWorkflow(timeout=None)

if __name__ == "__main__":
    import asyncio
    from argparse import ArgumentParser

    parser = ArgumentParser()
    parser.add_argument(
        "-p", "--path", required=True, help="Path to the invoice to extract"
    )
    parser.add_argument(
        "-m",
        "--mode",
        required=True,
        help="Extraction mode",
        choices=["base", "advanced", "premium"],
    )
    args = parser.parse_args()

    if not os.getenv("LLAMA_CLOUD_API_KEY", None):
        raise ValueError(
            "You need to set LLAMA_CLOUD_API_KEY in your environment before using this workflow"
        )

    asyncio.run(main(path=args.path, extraction_mode=args.mode))
