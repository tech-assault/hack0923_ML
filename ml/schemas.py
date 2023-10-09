from pydantic import BaseModel


class Data(BaseModel):
    """ Схема данных продукта перед предсказанием модели. """
    test: int
    strtest: str

