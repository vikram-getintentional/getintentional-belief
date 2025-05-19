from backend.utils.graph_base.debug_build_from_openai_result import save_graph_from_openai_result

mock_openai_output = [
    {
        "persona": {
            "title": "VP of Sales",
            "department": "Sales",
            "seniority": "Senior Management"
        },
        "job": "Forecast pipeline and manage sales targets",
        "pain": "CRM updates are inconsistent and cause forecasting errors",
        "dependencies": [
            {
                "dependent_persona": {
                    "title": "Head of RevOps",
                    "department": "Revenue Operations",
                    "seniority": "Mid Management"
                },
                "dependent_job": "Build revenue dashboards",
                "dependent_pain": "Unable to generate accurate dashboards due to poor pipeline visibility"
            }
        ]
    }
]

save_graph_from_openai_result(mock_openai_output)
print("✅ Graph built from mock OpenAI output")
