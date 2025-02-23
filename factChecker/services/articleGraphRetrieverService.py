from openai import OpenAI
from pathlib import Path
from langchain_community.graphs import Neo4jGraph
import os
import re
import json
from typing import List, Dict, Any
from django.core.management.base import CommandError
from neo4j import GraphDatabase
from dotenv import load_dotenv
from factChecker.services.entityExtractor import EntityExtractor
from factChecker.models import Article

class ArticleGraphRetrieverService:
    def __init__(self):
        load_dotenv()
        self.uri = os.getenv('NEO4J_URI')
        self.username = os.getenv('NEO4J_USERNAME')
        self.password = os.getenv('NEO4J_PASSWORD')
        self.driver = None
        
        self.graph = Neo4jGraph()
        
    def get_knowledge_graph_data(self, query: str) -> Dict[str, Any]:
        self.connect_to_neo4j()
        
        entity_extractor = EntityExtractor()
        query_structure = entity_extractor.get_query_structure_llm(query)
        
        entities = query_structure["entities"]
        relationships = query_structure["relationships"]
        
        if len(entities) == 0:
            raise CommandError(f'No entities found in query: {query}')
        
        data = {}
        data['main_entity_relations'] = self.get_entity_relations(entities[0], relationships)
        data['main_entity_articles'] = self.get_entity_articles(entities[0])
        data['shared_articles'] = []
        
        if len(entities) > 1:
            data['shared_articles'] = self.get_multiple_entity_shared_articles(entities)

        self.close_neo4j()

        return data

    def connect_to_neo4j(self):
        try:
            self.driver = GraphDatabase.driver(
                self.uri,
                auth=(self.username, self.password)
            )
            self.driver.verify_connectivity()
        except Exception as e:
            raise CommandError(f'Failed to connect to Neo4j: {str(e)}')
        
    def get_entity_relations(self, entity, relations, limit = 20):
        with self.driver.session() as session:
            query = """
                MATCH (p {id: $entity_id})-[r]-()
                WHERE type(r) <> "MENTIONS"
                WITH p, collect(r) AS allRelationships, $prioritizedRelations AS prioritizedTypes
                WITH p, allRelationships, prioritizedTypes,
                    [rel IN allRelationships WHERE type(rel) IN prioritizedTypes | rel] AS prioritizedRelationships
                UNWIND CASE WHEN size(prioritizedRelationships) > 0 THEN prioritizedRelationships ELSE allRelationships END AS selectedRelationship
                RETURN
                    p.id AS Source,
                    type(selectedRelationship) AS Relationship,
                    endNode(selectedRelationship).id AS Target
                LIMIT $limit
            """
            result = session.run(query, entity_id=entity, prioritizedRelations=relations, limit=limit)
            
            
            
            return [record["Source"] + '->' + record["Relationship"] + '->' + record["Target"] for record in result]
        
    def get_entity_articles(self, entity, limit = 5):
        with self.driver.session() as session:
            query = """
                MATCH (entity {id: $entity_id})-[r]-(doc:Document)
                RETURN DISTINCT doc.source
                LIMIT $limit
            """
            result = session.run(query, entity_id=entity, limit=limit)
            
            articles = []
            for record in result:
                article_id = int(record["doc.source"].split("_")[1])
                article = Article.objects.get(id=article_id)
                articles.append(article)

        return articles
    
    def get_multiple_entity_shared_articles(self, entities, limit = 5):
        with self.driver.session() as session:
            query = """
                MATCH (doc:Document)
                WHERE SIZE([entity_id IN $entity_ids 
                    WHERE EXISTS((doc)-[]->({id: entity_id}))]) >= 2
                RETURN DISTINCT doc.source
            """
            result = session.run(query, entity_ids=entities, limit=limit)
            
            articles = []
            for record in result:
                article_id = int(record["doc.source"].split("_")[1])
                article = Article.objects.get(id=article_id)
                articles.append(article)

        return articles

    def close_neo4j(self):
        if self.driver:
            self.driver.close()