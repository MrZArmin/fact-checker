from django.core.management.base import BaseCommand
from factChecker.models import Article, Benchmark, GraphBenchmark
from dotenv import load_dotenv
from openai import OpenAI
from pathlib import Path
import os
import json
import random

class Command(BaseCommand):
    help = 'Creates test database after clearing existing data. Use --graph option to load articles from JSON.'

    def add_arguments(self, parser):
        parser.add_argument(
            '--graph',
            action='store_true',
            help='Load article IDs from graph.json file',
        )

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        load_dotenv()
        self.prompts_dir = Path(__file__).parent.parent.parent / "prompts"
        self.client = OpenAI(api_key="")
        self.model = "gpt-4"
        self.system_prompt = self._load_prompt("create_questions_prompt.txt")
        self.temperature = 0.7

    def handle(self, *args, **kwargs):
        if kwargs['graph']:
            try:
                articles = self._load_articles_from_json().order_by('?')[:100]
                self._create_test(articles, True)
            except Exception as e:
                self.stdout.write(self.style.ERROR(f'Error loading graph.json: {str(e)}'))
                return
        else:
            articles = Article.objects.order_by('?')[:100]
            self._create_test(articles)


        self.stdout.write(self.style.SUCCESS('Successfully created new benchmark data'))

    def _load_articles_from_json(self):
        """Load article IDs from graph.json and return corresponding Article objects."""
        json_path = Path(__file__).parent.parent.parent.parent / "article_graph_progress.json"

        try:
            with open(json_path, 'r', encoding='utf-8') as f:
                graph_data = list(json.load(f))

            # Get Article objects for the extracted IDs
            articles = Article.objects.filter(id__in=graph_data)

            if not articles.exists():
                raise ValueError("No articles found with the provided IDs")

            return articles

        except FileNotFoundError:
            raise FileNotFoundError(f"graph.json not found at {json_path}")
        except json.JSONDecodeError:
            raise ValueError("Invalid JSON format in graph.json")

    def _create_test(self, articles, isGraph=False):
        if not isGraph:
            model = Benchmark
        else:
            model = GraphBenchmark
        for article in articles:
            text = article.title + " " + article.lead + " " + article.text
            answer_str = self._create_question_for_article(text)
            try:
                answer_dict = json.loads(answer_str)
                for answer in answer_dict:
                    model.objects.create(
                        article=article,
                        question=answer['question'],
                        answer=answer['answer']
                    )
            except json.JSONDecodeError as e:
                self.stdout.write(self.style.ERROR(f'Error parsing JSON for article {str(e)}'))
            except Exception as e:
                self.stdout.write(self.style.ERROR(f'Error creating benchmark for article {str(e)}'))

    def _create_question_for_article(self, article_text):
        try:
            messages = [
                {
                    "role": "system",
                    "content": self.system_prompt
                },
                {
                    "role": "user",
                    "content": f"Here is the article: {article_text}"
                }
            ]
            completion_params = {
                "messages": messages,
                "model": self.model,
                "temperature": self.temperature,
            }
            chat_completion = self.client.chat.completions.create(**completion_params)
            return chat_completion.choices[0].message.content
        except Exception as e:
            error_msg = f"Error generating response: {str(e)}"
            raise RuntimeError(error_msg)

    def _load_prompt(self, filename: str) -> str:
        """Load prompt from file."""
        try:
            prompt_path = self.prompts_dir / filename
            with open(prompt_path, "r", encoding="utf-8") as f:
                return f.read().strip()
        except Exception as e:
            raise FileNotFoundError(f"Could not load prompt {filename}: {str(e)}")
