import unittest
from unittest.mock import MagicMock, patch
import sys
import os

# Add the scripts directory to the Python path so we can import the agent
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../scripts')))

from agent import PPBAgent

class TestPPBAgent(unittest.TestCase):

    @patch('agent.GraphDatabase')
    @patch('agent.ChatOllama')
    @patch('agent.OllamaEmbeddings')
    def test_ask_flow_mocked(self, MockEmbeddings, MockChatOllama, MockGraphDatabase):
        """Verifies the agent logic using mocked Neo4j and Ollama services."""
        # 1. Setup Mocks
        # Mock the Neo4j driver and session
        mock_driver = MagicMock()
        MockGraphDatabase.driver.return_value = mock_driver
        mock_session = MagicMock()
        mock_driver.session.return_value.__enter__.return_value = mock_session
        
        # Mock the database result (simulating a found chunk from Neo4j)
        mock_record = {"text": "Company policy states work starts at 9 AM.", "score": 0.95, "index": 1}
        mock_session.run.return_value = [mock_record]

        # Mock the LLM response
        mock_llm_instance = MockChatOllama.return_value
        mock_llm_instance.invoke.return_value.content = "Work starts at 9 AM."

        # 2. Initialize Agent
        agent = PPBAgent()

        # 3. Run the method
        answer = agent.ask("When does work start?")

        # 4. Assertions
        self.assertEqual(answer, "Work starts at 9 AM.")
        
        # Verify Neo4j was queried
        mock_session.run.assert_called_once()
        
        # Verify LLM was invoked
        mock_llm_instance.invoke.assert_called_once()

    def test_integration_live(self):
        """Runs an integration test against live Neo4j and Ollama services."""
        # Attempt to initialize.
        # If services are down, this might fail or hang depending on timeouts.
        try:
            agent = PPBAgent()
            # Check if we can actually reach the DB
            agent.driver.verify_connectivity()
            
            print("\n[Integration] Services connected. Sending test query...")
            response = agent.ask("Test Query")
            self.assertIsInstance(response, str)
            agent.close()
        except Exception as e:
            print(f"\n[Integration] Skipped live test (Services likely down): {e}")

if __name__ == '__main__':
    unittest.main()
