from pydantic import BaseModel


class Employee(BaseModel):
    satisfaction_level: float
    last_evaluation: float
    number_project: int
    average_montly_hours: float
    time_spend_company: float
    Work_accident: int
    promotion_last_5years: int
    department: int
    salary_level: int
