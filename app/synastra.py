# -*- coding: utf-8 -*-
""" Synastra application with API Call """

import json

from enum import Enum
from typing import Any, Dict, List, Optional

from openai.types.chat import ChatCompletionFunctionToolParam
from pydantic import BaseModel, Field

from app.const import MISSING
from app.database import db
from app.exceptions import ConfigurationException
from app.logging_config import get_logger, setup_logging
from app.utils import color_text

setup_logging()
logger = get_logger(__name__)

class BirthData(BaseModel):
    year: int = Field(..., description="Birth year, for example 1997.")
    month: int = Field(..., ge=1, le=12, description="Birth month as a number from 1 to 12.")
    city: str = Field(..., description="Birth city, for example Winnipeg.")
    country_code: str = Field(..., min_length=2, max_length=2, description="Two-letter country code, for example CA or GB.")

class SynastrySubject(BaseModel):
    name: Optional[str] = Field(None, description="Optional name or label for the person.")
    birth_data: BirthData

class GetSynastryReportParams(BaseModel):
    subject_a: SynastrySubject
    subject_b: SynastrySubject

class Synastra:
    def tool_factory_get_synastry_report(self) -> ChatCompletionFunctionToolParam:
        schema = GetSynastryReportParams.model_json_schema()

        return ChatCompletionFunctionToolParam(
            type="function",
            function={
                "name": "get_synastry_report",
                "description": (
                    "Return a symbolic synastry relationship compatibility report for exactly two people. "
                    "Each person must include birth year, birth month, birth city, and two-letter country code. "
                    "Use this tool before answering relationship compatibility questions."
                ),
                "parameters": schema,
            },
        )

    def get_synastry_report(self, subject_a: dict, subject_b: dict) -> str:
        try:
            # Validate arguments using the same schema exposed to the model.
            params = GetSynastryReportParams(
                subject_a=subject_a,
                subject_b=subject_b,
            )
        except Exception as error:
            return json.dumps(
                {
                    "ok": False,
                    "error_type": "validation_error",
                    "message": "The tool call arguments were invalid.",
                    "details": str(error),
                }
            )

        return json.dumps(
            {
                "ok": True,
                "data": {
                    "subjects": {
                        "subject_a": params.subject_a.model_dump(),
                        "subject_b": params.subject_b.model_dump(),
                    },
                    "life_area_compatibility": [
                        {
                            "area": "Independence & Freedom",
                            "compatibility_score": 0.7755,
                            "description": (
                                "Balancing individual needs with partnership, maintaining personal space and growth. "
                                "Good compatibility - this area supports your relationship well."
                            ),
                            "key_factors": [
                                "Uranus placement",
                                "Aquarius emphasis",
                                "11th house emphasis",
                                "Mars-Uranus aspects",
                            ],
                        },
                        {
                            "area": "Communication & Understanding",
                            "compatibility_score": 0.8498,
                            "description": (
                                "How well you communicate, share ideas, and understand each other's thoughts and perspectives. "
                                "Excellent compatibility - you naturally harmonize in this area."
                            ),
                            "key_factors": [
                                "Mercury placement",
                                "Mercury aspects",
                                "3rd house emphasis",
                                "Air sign emphasis",
                            ],
                        },
                    ],
                },
            }
        )


synastra_app = Synastra()