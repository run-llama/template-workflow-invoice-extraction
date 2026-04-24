import pytest
from invoice_extraction.workflow import (
    FeedbackRequiredEvent,
    HumanFeedbackEvent,
    workflow,
)
from llama_cloud_fake import FakeLlamaCloudServer


@pytest.mark.parametrize("extraction_mode", ["base", "advanced", "premium"])
async def test_invoice_extraction_workflow(
    monkeypatch: pytest.MonkeyPatch,
    fake: FakeLlamaCloudServer,
    extraction_mode: str,
) -> None:
    """Exercise files.create + extract.create + wait_for_completion via the fake,
    then approve through the human-in-the-loop step."""
    monkeypatch.setenv("LLAMA_CLOUD_API_KEY", "fake-api-key")
    handler = workflow.run(
        path="tests/files/test.pdf",
        extraction_mode=extraction_mode,
    )
    feedback_event: FeedbackRequiredEvent | None = None
    async for ev in handler.stream_events():
        if isinstance(ev, FeedbackRequiredEvent):
            feedback_event = ev
            handler.ctx.send_event(HumanFeedbackEvent(approved=True))
            break
    result = await handler
    assert feedback_event is not None
    assert isinstance(feedback_event.extraction_result, str)
    assert result == feedback_event.extraction_result
