from pydantic import BaseModel
from typing import Optional

class EmailProcessRequest(BaseModel):
    text: Optional[str] = None

class EmailProcessResponse(BaseModel):
    category: str
    suggested_response: str
