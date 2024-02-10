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

    # satisfaction_level: float
    # last_evaluation: float
    # number_project: int
    # average_montly_hours: float
    # time_spend_company: float
    # Work_accident: int
    # promotion_last_5years: int
    # department: int
    # salary_level: int
