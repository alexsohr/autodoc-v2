"""Chat step definitions for BDD tests

This module contains step definitions for chat workflow scenarios
including session management, asking questions, and conversation history.
"""

import pytest
from pytest_bdd import given, when, then, parsers


# =============================================================================
# Setup Steps
# =============================================================================

@given("I have an analyzed repository")
def have_analyzed_repository(client, context):
    """Ensure we have a repository that's ready for chat queries."""
    if "repository_id" not in context:
        # Create and register a repository
        payload = {"url": "https://github.com/test-org/chat-test-repo"}
        response = client.post("/repositories", json=payload)
        
        if response.status_code == 201:
            repo_id = response.json()["id"]
            context["repository_id"] = repo_id
            context["repository"] = response.json()
            # Track for cleanup
            context["created_repositories"].append(repo_id)
        elif response.status_code == 409:
            # Repository exists, get its ID
            list_response = client.get("/repositories")
            if list_response.status_code == 200:
                repos = list_response.json().get("repositories", [])
                for repo in repos:
                    if "chat-test-repo" in repo.get("url", ""):
                        context["repository_id"] = repo["id"]
                        context["repository"] = repo
                        break


# =============================================================================
# Chat Session Management Steps
# =============================================================================

@when("I create a chat session for the repository")
def create_chat_session(client, context):
    """Create a new chat session for the current repository."""
    repository_id = context.get("repository_id")
    assert repository_id is not None, "No repository_id in context"
    
    response = client.post(f"/repositories/{repository_id}/chat/sessions")
    context["response"] = response
    
    if response.status_code == 201:
        session_id = response.json()["id"]
        context["session_id"] = session_id
        context["session"] = response.json()
        # Track for cleanup
        context["created_sessions"].append({
            "repository_id": repository_id,
            "session_id": session_id
        })


@given("I have a chat session")
def have_chat_session(client, context):
    """Ensure a chat session exists for the repository."""
    if "session_id" not in context:
        repository_id = context.get("repository_id")
        assert repository_id is not None, "No repository_id in context"
        
        response = client.post(f"/repositories/{repository_id}/chat/sessions")
        
        if response.status_code == 201:
            session_id = response.json()["id"]
            context["session_id"] = session_id
            context["session"] = response.json()
            # Track for cleanup
            context["created_sessions"].append({
                "repository_id": repository_id,
                "session_id": session_id
            })


@when("I list chat sessions for the repository")
def list_chat_sessions(client, context):
    """List all chat sessions for the current repository."""
    repository_id = context.get("repository_id")
    assert repository_id is not None, "No repository_id in context"
    
    response = client.get(f"/repositories/{repository_id}/chat/sessions")
    context["response"] = response


@when(parsers.parse('I list chat sessions with status filter "{status}"'))
def list_chat_sessions_with_status(client, context, status: str):
    """List chat sessions filtered by status."""
    repository_id = context.get("repository_id")
    assert repository_id is not None, "No repository_id in context"
    
    response = client.get(
        f"/repositories/{repository_id}/chat/sessions",
        params={"status": status}
    )
    context["response"] = response


@when("I get the chat session details")
def get_chat_session_details(client, context):
    """Get details of the current chat session."""
    repository_id = context.get("repository_id")
    session_id = context.get("session_id")
    assert repository_id is not None, "No repository_id in context"
    assert session_id is not None, "No session_id in context"
    
    response = client.get(
        f"/repositories/{repository_id}/chat/sessions/{session_id}"
    )
    context["response"] = response


@when(parsers.parse('I get chat session with ID "{session_id}"'))
def get_chat_session_by_id(client, context, session_id: str):
    """Get details of a chat session by its ID."""
    repository_id = context.get("repository_id")
    assert repository_id is not None, "No repository_id in context"
    
    response = client.get(
        f"/repositories/{repository_id}/chat/sessions/{session_id}"
    )
    context["response"] = response


@when("I delete the chat session")
def delete_chat_session(client, context):
    """Delete the current chat session."""
    repository_id = context.get("repository_id")
    session_id = context.get("session_id")
    assert repository_id is not None, "No repository_id in context"
    assert session_id is not None, "No session_id in context"
    
    response = client.delete(
        f"/repositories/{repository_id}/chat/sessions/{session_id}"
    )
    context["response"] = response
    context["deleted_session_id"] = session_id


@then("the chat session should no longer exist")
def chat_session_should_not_exist(client, context):
    """Verify the deleted chat session no longer exists."""
    repository_id = context.get("repository_id")
    session_id = context.get("deleted_session_id")
    assert repository_id is not None, "No repository_id in context"
    assert session_id is not None, "No deleted_session_id in context"
    
    response = client.get(
        f"/repositories/{repository_id}/chat/sessions/{session_id}"
    )
    assert response.status_code == 404, (
        f"Expected 404 for deleted session, got {response.status_code}"
    )


@then(parsers.parse('all sessions in the response should have status "{status}"'))
def all_sessions_should_have_status(context, status: str):
    """Assert all sessions in the response have the expected status."""
    response = context.get("response")
    assert response is not None, "No response in context"
    
    data = response.json()
    sessions = data.get("sessions", [])
    
    for session in sessions:
        assert session["status"] == status, (
            f"Session {session['id']} has status '{session['status']}', "
            f"expected '{status}'"
        )


# =============================================================================
# Asking Questions Steps
# =============================================================================

@when(parsers.parse('I ask "{question}"'))
def ask_question(client, context, question: str):
    """Ask a question in the current chat session."""
    repository_id = context.get("repository_id")
    session_id = context.get("session_id")
    assert repository_id is not None, "No repository_id in context"
    assert session_id is not None, "No session_id in context"
    
    payload = {"content": question}
    response = client.post(
        f"/repositories/{repository_id}/chat/sessions/{session_id}/questions",
        json=payload
    )
    context["response"] = response
    
    if response.status_code == 201:
        context["last_question"] = response.json().get("question")
        context["last_answer"] = response.json().get("answer")
        context["questions_asked"] = context.get("questions_asked", 0) + 1


@when(parsers.parse('I ask "{question}" with context hint "{hint}"'))
def ask_question_with_hint(client, context, question: str, hint: str):
    """Ask a question with a context hint."""
    repository_id = context.get("repository_id")
    session_id = context.get("session_id")
    assert repository_id is not None, "No repository_id in context"
    assert session_id is not None, "No session_id in context"
    
    payload = {"content": question, "context_hint": hint}
    response = client.post(
        f"/repositories/{repository_id}/chat/sessions/{session_id}/questions",
        json=payload
    )
    context["response"] = response
    
    if response.status_code == 201:
        context["last_question"] = response.json().get("question")
        context["last_answer"] = response.json().get("answer")
        context["questions_asked"] = context.get("questions_asked", 0) + 1


@given("I have asked a question")
def have_asked_question(client, context):
    """Ensure at least one question has been asked in the session."""
    if context.get("questions_asked", 0) == 0:
        repository_id = context.get("repository_id")
        session_id = context.get("session_id")
        
        payload = {"content": "What does this project do?"}
        response = client.post(
            f"/repositories/{repository_id}/chat/sessions/{session_id}/questions",
            json=payload
        )
        
        if response.status_code == 201:
            context["last_question"] = response.json().get("question")
            context["last_answer"] = response.json().get("answer")
            context["questions_asked"] = 1


@given("I have asked multiple questions")
def have_asked_multiple_questions(client, context):
    """Ensure multiple questions have been asked in the session."""
    questions = [
        "What is this project about?",
        "How do I install it?",
        "What are the main features?",
    ]
    
    repository_id = context.get("repository_id")
    session_id = context.get("session_id")
    
    for question in questions:
        if context.get("questions_asked", 0) < 3:
            payload = {"content": question}
            response = client.post(
                f"/repositories/{repository_id}/chat/sessions/{session_id}/questions",
                json=payload
            )
            
            if response.status_code == 201:
                context["questions_asked"] = context.get("questions_asked", 0) + 1


# =============================================================================
# Conversation History Steps
# =============================================================================

@when("I get the conversation history")
def get_conversation_history(client, context):
    """Get the conversation history for the current session."""
    repository_id = context.get("repository_id")
    session_id = context.get("session_id")
    assert repository_id is not None, "No repository_id in context"
    assert session_id is not None, "No session_id in context"
    
    response = client.get(
        f"/repositories/{repository_id}/chat/sessions/{session_id}/history"
    )
    context["response"] = response


@when(parsers.parse("I get the conversation history with limit {limit:d}"))
def get_conversation_history_with_limit(client, context, limit: int):
    """Get the conversation history with pagination limit."""
    repository_id = context.get("repository_id")
    session_id = context.get("session_id")
    assert repository_id is not None, "No repository_id in context"
    assert session_id is not None, "No session_id in context"
    
    response = client.get(
        f"/repositories/{repository_id}/chat/sessions/{session_id}/history",
        params={"limit": limit}
    )
    context["response"] = response


@then(parsers.parse("the history should have at most {count:d} items"))
def history_should_have_at_most_items(context, count: int):
    """Assert history has at most the specified number of items."""
    response = context.get("response")
    assert response is not None, "No response in context"
    
    data = response.json()
    qa_list = data.get("questions_and_answers", [])
    
    assert len(qa_list) <= count, (
        f"Expected at most {count} items, got {len(qa_list)}"
    )


# =============================================================================
# Streaming Steps
# =============================================================================

@when("I access the streaming endpoint")
def access_streaming_endpoint(client, context):
    """Access the SSE streaming endpoint."""
    repository_id = context.get("repository_id")
    session_id = context.get("session_id")
    assert repository_id is not None, "No repository_id in context"
    assert session_id is not None, "No session_id in context"
    
    response = client.get(
        f"/repositories/{repository_id}/chat/sessions/{session_id}/stream"
    )
    context["response"] = response


@then(parsers.parse('the response content type should be "{content_type}"'))
def response_content_type_should_be(context, content_type: str):
    """Assert the response has the expected content type."""
    response = context.get("response")
    assert response is not None, "No response in context"
    
    actual_content_type = response.headers.get("content-type", "")
    assert content_type in actual_content_type, (
        f"Expected content type containing '{content_type}', "
        f"got '{actual_content_type}'"
    )


# =============================================================================
# Session Property Assertions
# =============================================================================

@then(parsers.parse('the session status should be "{status}"'))
def session_status_should_be(context, status: str):
    """Assert the session has the expected status."""
    response = context.get("response")
    assert response is not None, "No response in context"
    
    data = response.json()
    assert data.get("status") == status, (
        f"Expected status '{status}', got '{data.get('status')}'"
    )


@then(parsers.parse("the session message count should be {count:d}"))
def session_message_count_should_be(context, count: int):
    """Assert the session has the expected message count."""
    response = context.get("response")
    assert response is not None, "No response in context"
    
    data = response.json()
    assert data.get("message_count") == count, (
        f"Expected message count {count}, got {data.get('message_count')}"
    )


@then(parsers.parse("the session should have at least {count:d} messages"))
def session_should_have_at_least_messages(client, context, count: int):
    """Assert the session has at least the specified number of messages."""
    repository_id = context.get("repository_id")
    session_id = context.get("session_id")
    
    response = client.get(
        f"/repositories/{repository_id}/chat/sessions/{session_id}"
    )
    data = response.json()
    
    assert data.get("message_count", 0) >= count, (
        f"Expected at least {count} messages, got {data.get('message_count')}"
    )


# =============================================================================
# Answer Property Assertions
# =============================================================================

@then("the answer should have content")
def answer_should_have_content(context):
    """Assert the answer has non-empty content."""
    response = context.get("response")
    assert response is not None, "No response in context"
    
    data = response.json()
    answer = data.get("answer", {})
    content = answer.get("content", "")
    
    assert len(content) > 0, "Answer content should not be empty"


@then("the answer should have a confidence score")
def answer_should_have_confidence_score(context):
    """Assert the answer has a confidence score."""
    response = context.get("response")
    assert response is not None, "No response in context"
    
    data = response.json()
    answer = data.get("answer", {})
    
    assert "confidence_score" in answer, "Answer should have confidence_score"


@then("the confidence score should be between 0 and 1")
def confidence_score_should_be_valid(context):
    """Assert the confidence score is a valid value between 0 and 1."""
    response = context.get("response")
    assert response is not None, "No response in context"
    
    data = response.json()
    answer = data.get("answer", {})
    score = answer.get("confidence_score")
    
    assert score is not None, "No confidence_score in answer"
    assert 0.0 <= score <= 1.0, (
        f"Confidence score should be between 0 and 1, got {score}"
    )


@then("the answer should include citations")
def answer_should_include_citations(context):
    """Assert the answer includes code citations."""
    response = context.get("response")
    assert response is not None, "No response in context"
    
    data = response.json()
    answer = data.get("answer", {})
    citations = answer.get("citations", [])
    
    assert "citations" in answer, "Answer should have citations field"


@then("the answer should include generation time")
def answer_should_include_generation_time(context):
    """Assert the answer includes generation time."""
    response = context.get("response")
    assert response is not None, "No response in context"
    
    data = response.json()
    answer = data.get("answer", {})
    
    assert "generation_time" in answer, "Answer should have generation_time"
    assert answer["generation_time"] >= 0, "Generation time should be non-negative"


@then("the question should include context files")
def question_should_include_context_files(context):
    """Assert the question includes context files."""
    response = context.get("response")
    assert response is not None, "No response in context"
    
    data = response.json()
    question = data.get("question", {})
    
    assert "context_files" in question, "Question should have context_files"

