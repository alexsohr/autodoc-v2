Feature: Chat Query Workflow
  As an API user
  I want to ask questions about a repository codebase
  So that I can understand how the code works

  Background:
    Given I am authenticated as an admin user
    And I have an analyzed repository

  # ==========================================================================
  # Chat Session Management
  # ==========================================================================

  Scenario: Create a new chat session
    When I create a chat session for the repository
    Then the response status should be 201
    And the response should have a valid UUID id
    And the session status should be "active"
    And the session message count should be 0

  Scenario: List chat sessions for a repository
    Given I have a chat session
    When I list chat sessions for the repository
    Then the response status should be 200
    And the response should contain "sessions"
    And the response should contain "total"

  Scenario: List active chat sessions only
    Given I have a chat session
    When I list chat sessions with status filter "active"
    Then the response status should be 200
    And all sessions in the response should have status "active"

  Scenario: Get chat session details
    Given I have a chat session
    When I get the chat session details
    Then the response status should be 200
    And the response should contain "id"
    And the response should contain "repository_id"
    And the response should contain "status"
    And the response should contain "message_count"

  Scenario: Fail to get non-existent chat session
    When I get chat session with ID "00000000-0000-0000-0000-000000000000"
    Then the response status should be 404

  Scenario: Delete a chat session
    Given I have a chat session
    When I delete the chat session
    Then the response status should be 204
    And the chat session should no longer exist

  # ==========================================================================
  # Asking Questions
  # ==========================================================================

  Scenario: Ask a question about the codebase
    Given I have a chat session
    When I ask "How does authentication work in this application?"
    Then the response status should be 201
    And the response should contain "question"
    And the response should contain "answer"
    And the answer should have content
    And the answer should have a confidence score

  Scenario: Ask a question with context hint
    Given I have a chat session
    When I ask "How do I configure the database?" with context hint "database, config"
    Then the response status should be 201
    And the question should include context files

  Scenario: Fail to ask empty question
    Given I have a chat session
    When I ask ""
    Then the response status should be 400

  Scenario: Ask multiple questions in a session
    Given I have a chat session
    When I ask "What is the main purpose of this project?"
    Then the response status should be 201
    When I ask "How do I run the tests?"
    Then the response status should be 201
    And the session should have at least 2 messages

  # ==========================================================================
  # Conversation History
  # ==========================================================================

  Scenario: Get conversation history
    Given I have a chat session
    And I have asked a question
    When I get the conversation history
    Then the response status should be 200
    And the response should contain "session_id"
    And the response should contain "questions_and_answers"
    And the response should contain "total"

  Scenario: Get conversation history with pagination
    Given I have a chat session
    And I have asked multiple questions
    When I get the conversation history with limit 2
    Then the response status should be 200
    And the history should have at most 2 items

  # ==========================================================================
  # Answer Quality
  # ==========================================================================

  Scenario: Answer includes source code citations
    Given I have a chat session
    When I ask "How does the user authentication middleware work?"
    Then the response status should be 201
    And the answer should include citations

  Scenario: Answer has reasonable confidence score
    Given I have a chat session
    When I ask "What testing patterns are used in this codebase?"
    Then the response status should be 201
    And the confidence score should be between 0 and 1

  Scenario: Answer includes generation time
    Given I have a chat session
    When I ask "What are the main dependencies?"
    Then the response status should be 201
    And the answer should include generation time

  # ==========================================================================
  # Streaming Responses
  # ==========================================================================

  Scenario: Access streaming endpoint
    Given I have a chat session
    When I access the streaming endpoint
    Then the response status should be 200
    And the response content type should be "text/event-stream"

