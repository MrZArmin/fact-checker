from django.core.management.base import BaseCommand
from langchain_core.runnables import (
    RunnableBranch,
    RunnableLambda,
    RunnableParallel,
    RunnablePassthrough,
)
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.prompts.prompt import PromptTemplate
from langchain_core.pydantic_v1 import BaseModel, Field
from typing import Tuple, List, Optional
from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.output_parsers import StrOutputParser
import os
from langchain_community.graphs import Neo4jGraph
from langchain.text_splitter import TokenTextSplitter
from langchain_openai import ChatOpenAI
from langchain_experimental.graph_transformers import LLMGraphTransformer
from neo4j import GraphDatabase
from langchain_community.vectorstores import Neo4jVector
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores.neo4j_vector import remove_lucene_chars
from langchain_core.runnables import ConfigurableField, RunnableParallel, RunnablePassthrough
from langchain_core.documents import Document
from factChecker.models import Article

class Command(BaseCommand):
    help = 'Create a knowledge graph from the given data'
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        
    def handle(self, *args, **options):        
        # Initialize Neo4j Graph
        graph = Neo4jGraph()
        
        llm = ChatOpenAI(temperature=0, model_name="gpt-4o")
        text_splitter = TokenTextSplitter(chunk_size=512, chunk_overlap=24)
        llm_transformer = LLMGraphTransformer(llm=llm)
        
        # Process articles one by one
        article_ids = [37614, 15715]
        
        for article_id in article_ids:
            try:
                # Get single article
                article = Article.objects.get(id=article_id)
                
                # Create document from article
                doc = Document(
                    page_content=article.title + " " + article.lead + " " + article.text,
                    metadata={"source": f"article_{article_id}"}
                )
                
                # Split document
                documents = text_splitter.split_documents([doc])
                print(f"Processing article {article_id} - Split into {len(documents)} chunks")
                
                # Convert to graph documents
                graph_documents = llm_transformer.convert_to_graph_documents(documents)
                
                # Add to graph
                graph.add_graph_documents(
                    graph_documents,
                    baseEntityLabel=True,
                    include_source=True
                )
                
                self.stdout.write(
                    self.style.SUCCESS(f'Successfully processed article {article_id}')
                )
                
            except Article.DoesNotExist:
                self.stdout.write(
                    self.style.WARNING(f'Article {article_id} not found')
                )
            except Exception as e:
                self.stdout.write(
                    self.style.ERROR(f'Error processing article {article_id}: {str(e)}')
                )
            
        
        self.stdout.write(self.style.SUCCESS('Finished processing all articles'))