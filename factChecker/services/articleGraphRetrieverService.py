from openai import OpenAI
from pathlib import Path
from langchain_community.graphs import Neo4jGraph
import os
import re

class ArticleGraphRetrieverService():
    def __init__(self):
        self.api_key = os.getenv('OPENAI_API_KEY')
        self.client = OpenAI(api_key=self.api_key)
        self.openai_model = "gpt-4o"
        self.temperature = 0.7
        self.prompts_dir = Path(__file__).parent.parent / "prompts"

    def findSimilarNodes(self, user_query="Ki az az Orbán Viktor?"):
        """
        Collects the neighborhood of entities mentioned
        in the question
        """
        graph = Neo4jGraph()
        result = ""
        entities = self._getEntities(user_query)
        return
        for entity in entities.names:
            response = graph.query(
                """CALL db.index.fulltext.queryNodes('entity', $query, {limit:2})
                YIELD node,score
                CALL {
                MATCH (node)-[r:!MENTIONS]->(neighbor)
                RETURN node.id + ' - ' + type(r) + ' -> ' + neighbor.id AS output
                UNION
                MATCH (node)<-[r:!MENTIONS]-(neighbor)
                RETURN neighbor.id + ' - ' + type(r) + ' -> ' +  node.id AS output
                }
                RETURN output LIMIT 50
                """,
                {"query": self._generate_full_text_query(entity)},
            )
            result += "\n".join([el['output'] for el in response])

        return result

    # Ez nem mukodik, lehet spacyvel kéne, a dátumokra meg kitalálni valamit
    def _getEntities(self, user_query, systemPromptTxt="get_entities_prompt.txt"):
        """Identifying information about entities."""

        try:
            messages = [
                {
                    "role": "system",
                    "content": self._load_prompt(systemPromptTxt)
                },
                {
                    "role": "user",
                    "content": f"Use the given format to extract information from the following: {user_query}"
                }
            ]

            completion_params = {
                "messages": messages,
                "model": self.openai_model,
                "temperature": self.temperature,
            }

            chat_completion = self.client.chat.completions.create(**completion_params)
            print(chat_completion.choices[0].message.content)
            return chat_completion.choices[0].message.content

        except Exception as e:
            error_msg = f"Error generating response: {str(e)}"
            raise RuntimeError(error_msg)


    def _generate_full_text_query(self, input):
        """
        Generate a full-text search query for a given input string.
        """
        full_text_query = ""
        words = [el for el in self._remove_lucene_chars(input).split() if el]
        for word in words[:-1]:
            full_text_query += f" {word}~2 AND"
        full_text_query += f" {words[-1]}~2"

        return full_text_query.strip()

    def _remove_lucene_chars(self, input_string):
        """
        Remove or escape special characters that have special meaning in Lucene query syntax.
        """
        # Characters to remove entirely
        chars_to_remove = r'[+\-&|!(){}[\]^"~*?:]'

        # Use regex to remove special Lucene characters

        cleaned_string = re.sub(chars_to_remove, ' ', input_string)

        # Optional: Remove multiple consecutive whitespaces
        cleaned_string = re.sub(r'\s+', ' ', cleaned_string)

        return cleaned_string.strip()

    def _load_prompt(self, filename: str) -> str:
        """Load prompt from file."""
        try:
            prompt_path = self.prompts_dir / filename
            with open(prompt_path, "r", encoding="utf-8") as f:
                return f.read().strip()
        except Exception as e:
            raise FileNotFoundError(f"Could not load prompt {filename}: {str(e)}")


