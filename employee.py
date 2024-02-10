from pydantic import BaseModel


class Employee(BaseModel):
    firstname: str
    lastname: str
    department: int
    salary_level: int
    no_of_projects: int
    avg_monthly_hours: float
    time_spend_company: float
    promotions: int
    work_accidents: int
    job_satisfaction: float
    last_evaluation: float

    model_config = {
        "json_schema_extra": {
            "examples": [
                {
                    "firstname": "Sachin",
                    "lastname": "Lakshan",
                    "department": 3,
                    "salary_level": 0,
                    "no_of_projects": 3,
                    "avg_monthly_hours": 240,
                    "time_spend_company": 3,
                    "promotions": 0,
                    "work_accidents": 0,
                    "job_satisfaction": 60,
                    "last_evaluation": 52
                }
            ]
        }
    }
