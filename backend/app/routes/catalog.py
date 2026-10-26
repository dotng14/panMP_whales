from typing import Literal

from fastapi import APIRouter, Body, Query
from fastapi.responses import FileResponse

from app.services import catalog_service

router = APIRouter(prefix="/api/catalog")


@router.get("/wavs")
def get_wavs() -> dict:
    return {"wavs": catalog_service.list_wavs()}


@router.get("/wavs/{wav_id}/segmentations")
def get_wav_segmentations(wav_id: int) -> dict:
    return {"segmentations": catalog_service.list_gpl_segmentations_for_wav(wav_id)}


@router.get("/runs")
def get_runs() -> dict:
    return {"runs": catalog_service.list_runs()}


@router.get("/runs/{run_id}/templates")
def get_run_templates(run_id: int) -> dict:
    return {"templates": catalog_service.list_templates(run_id)}


@router.get("/runs/{run_id}/rounds")
def get_run_rounds(run_id: int) -> dict:
    """One row per outer pass of a randomized build."""
    return {"rounds": catalog_service.list_round_logs(run_id)}


@router.get("/runs/{run_id}/templates/{template_id}", response_model=None)
def get_run_template(
    run_id: int,
    template_id: int,
    include: str = Query("", description="Set to 'data' to download the .npy"),
):
    if include == "data":
        path, filename = catalog_service.get_template_npy(run_id, template_id)
        return FileResponse(path, media_type="application/octet-stream", filename=filename)
    return catalog_service.get_template(run_id, template_id)


@router.get("/dictionaries")
def get_dictionaries() -> dict:
    return {"dictionaries": catalog_service.list_dictionaries()}


@router.get("/dictionaries/{dictionary_id}")
def get_dictionary(dictionary_id: int) -> dict:
    return catalog_service.get_dictionary(dictionary_id)


@router.patch("/dictionaries/{dictionary_id}")
def patch_dictionary_label(dictionary_id: int, label: str | None = Body(None, embed=True)) -> dict:
    """Set or clear the user label (empty → unnamed)."""
    return catalog_service.set_dictionary_label(dictionary_id, label)


@router.get("/dictionaries/{dictionary_id}/templates")
def get_dictionary_templates(dictionary_id: int) -> dict:
    return {"templates": catalog_service.list_dictionary_templates(dictionary_id)}


@router.patch("/dictionaries/{dictionary_id}/templates/{template_id}")
def patch_dictionary_template_label(
    dictionary_id: int, template_id: int, label: str | None = Body(None, embed=True)
) -> dict:
    """Set or clear the user label of one atom within a dictionary."""
    return catalog_service.set_dictionary_template_label(dictionary_id, template_id, label)


@router.get("/dictionaries/{dictionary_id}/windows")
def get_dictionary_windows(dictionary_id: int) -> dict:
    return {"windows": catalog_service.list_dictionary_windows(dictionary_id)}


@router.get("/dictionaries/{dictionary_id}/lineage")
def get_dictionary_lineage(dictionary_id: int) -> dict:
    return {"lineage": catalog_service.get_dictionary_lineage(dictionary_id)}


@router.get("/dictionaries/{dictionary_id}/matchings")
def get_dictionary_matchings(
    dictionary_id: int,
    template_pk: int | None = Query(None),
    window_id: str | None = Query(None),
    feedback: Literal["good", "bad", "unlabeled"] | None = Query(None),
) -> dict:
    return {
        "matchings": catalog_service.list_dictionary_matchings(
            dictionary_id, template_pk=template_pk, window_id=window_id, feedback=feedback
        )
    }


@router.get("/dictionaries/{dictionary_id}/placements")
def get_dictionary_placements(dictionary_id: int) -> dict:
    """Placements in seconds from the WAV start, for the viewer's atom lane."""
    return catalog_service.list_dictionary_placements(dictionary_id)


@router.get("/matchings/{matching_id}")
def get_matching(matching_id: int) -> dict:
    return catalog_service.get_matching(matching_id)


@router.put("/matchings/{matching_id}/feedback")
def put_matching_feedback(
    matching_id: int,
    label: Literal["good", "bad"] = Body(..., embed=True),
    note: str | None = Body(None, embed=True),
) -> dict:
    return catalog_service.set_matching_feedback(matching_id, label, note=note)


@router.delete("/matchings/{matching_id}/feedback")
def delete_matching_feedback(matching_id: int) -> dict:
    return catalog_service.delete_matching_feedback(matching_id)


@router.get("/tool-runs")
def get_tool_runs(learning_run_id: int | None = Query(None)) -> dict:
    return {"toolRuns": catalog_service.list_tool_runs(learning_run_id=learning_run_id)}


@router.get("/tool-runs/{tool_run_id}")
def get_tool_run(tool_run_id: int) -> dict:
    return catalog_service.get_tool_run(tool_run_id)


@router.get("/csvs")
def get_csvs() -> dict:
    return {"csvs": catalog_service.list_csvs()}


@router.get("/csvs/{csv_id}")
def get_csv(csv_id: int) -> dict:
    return catalog_service.get_csv(csv_id)


@router.patch("/csvs/{csv_id}")
def patch_csv_label(csv_id: int, label: str | None = Body(None, embed=True)) -> dict:
    """Set or clear the user label (empty → unnamed)."""
    return catalog_service.set_csv_label(csv_id, label)


@router.post("/reindex")
def post_reindex() -> dict:
    """Rescan WAV_LIBRARY_ROOTS, CATALOG_RUNS_ROOTS and CATALOG_CSV_ROOTS."""
    return {"indexed": catalog_service.reindex()}
