from typing import List, Optional
from pydantic import BaseModel, Field

class EntityAttribute(BaseModel):
    key: str = Field(description="The name of the attribute (e.g., 'age', 'location', 'role')")
    value: str = Field(description="The value of the attribute")

class Entity(BaseModel):
    name: str = Field(description="The formal name of the entity, capitalized. e.g., 'John Doe', 'Microsoft', 'Paris'")
    type: str = Field(description="The type or category of the entity. E.g., 'PERSON', 'ORGANIZATION', 'LOCATION', 'EVENT', 'CONCEPT'")
    description: str = Field(description="A brief description or summary of what this entity is based on the text.")
    attributes: List[EntityAttribute] = Field(default_factory=list, description="Any key-value attributes associated with the entity.")

class Relationship(BaseModel):
    source_entity: str = Field(description="The exact name of the source entity as extracted.")
    target_entity: str = Field(description="The exact name of the target entity as extracted.")
    relationship_type: str = Field(description="The type of relationship between the source and target. Keep it concise, e.g., 'EMPLOYED_BY', 'LOCATED_IN', 'FOUNDED'.")
    description: str = Field(description="A brief explanation of why this relationship exists based on the source text.")

class GraphData(BaseModel):
    entities: List[Entity] = Field(description="The list of all distinct entities found in the text.")
    relationships: List[Relationship] = Field(description="The list of all relationships between the extracted entities.")
