from pydantic import BaseModel, Field


class SquadQuery(BaseModel):
    """
    PyDantic model of a query.
    """
    context: str or None = Field(default=None, title='The context.')
    question: str or None = Field(default=None, title='The question.')
    model: str or None = Field(default='Trained', title='The model. Trained or Pretrained.')


class SquadAnswer(BaseModel):
    """
    Pydantic model of an answer.
    """
    context: str or None = Field(default=None, title='The context.')
    question: str or None = Field(default=None, title='The question.')
    answer: str or None = Field(default=None, title='The answer.')
