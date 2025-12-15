"""Common step definitions shared across all BDD features

This module contains reusable steps for authentication, response validation,
and other common operations used across multiple feature files.
"""

from pytest_bdd import given, then, parsers
import pytest


# =============================================================================
# Authentication Steps
# =============================================================================

@given("I am authenticated as an admin user")
def authenticated_admin_user(context):
    """Set up authentication context for an admin user.
    
    Note: The current API uses mock authentication, so this step
    primarily sets up context for future auth implementation.
    """
    context["auth_token"] = "mock-admin-token"
    context["user_role"] = "admin"
    context["is_authenticated"] = True


@given("I am authenticated as a regular user")
def authenticated_regular_user(context):
    """Set up authentication context for a regular user."""
    context["auth_token"] = "mock-user-token"
    context["user_role"] = "user"
    context["is_authenticated"] = True


@given("I am not authenticated")
def not_authenticated(context):
    """Clear any authentication context."""
    context["auth_token"] = None
    context["user_role"] = None
    context["is_authenticated"] = False


# =============================================================================
# Response Status Assertions
# =============================================================================

@then(parsers.parse("the response status should be {status_code:d}"))
def response_status_should_be(context, status_code: int):
    """Assert that the response has the expected HTTP status code."""
    response = context.get("response")
    assert response is not None, "No response found in context"
    assert response.status_code == status_code, (
        f"Expected status {status_code}, got {response.status_code}. "
        f"Response body: {response.text}"
    )


@then("the response should be successful")
def response_should_be_successful(context):
    """Assert that the response has a 2xx status code."""
    response = context.get("response")
    assert response is not None, "No response found in context"
    assert 200 <= response.status_code < 300, (
        f"Expected success status (2xx), got {response.status_code}. "
        f"Response body: {response.text}"
    )


@then("the response should indicate an error")
def response_should_indicate_error(context):
    """Assert that the response has a 4xx or 5xx status code."""
    response = context.get("response")
    assert response is not None, "No response found in context"
    assert response.status_code >= 400, (
        f"Expected error status (4xx or 5xx), got {response.status_code}"
    )


# =============================================================================
# Response Body Assertions
# =============================================================================

@then(parsers.parse('the response should contain "{field}"'))
def response_should_contain_field(context, field: str):
    """Assert that the response JSON contains a specific field."""
    response = context.get("response")
    assert response is not None, "No response found in context"
    data = response.json()
    assert field in data, f"Field '{field}' not found in response: {data}"


@then(parsers.parse('the response "{field}" should be "{value}"'))
def response_field_should_be_string(context, field: str, value: str):
    """Assert that a response field equals a specific string value."""
    response = context.get("response")
    assert response is not None, "No response found in context"
    data = response.json()
    assert field in data, f"Field '{field}' not found in response: {data}"
    assert str(data[field]) == value, (
        f"Expected {field}='{value}', got '{data[field]}'"
    )


@then(parsers.parse('the response "{field}" should be {value:d}'))
def response_field_should_be_int(context, field: str, value: int):
    """Assert that a response field equals a specific integer value."""
    response = context.get("response")
    assert response is not None, "No response found in context"
    data = response.json()
    assert field in data, f"Field '{field}' not found in response: {data}"
    assert data[field] == value, f"Expected {field}={value}, got {data[field]}"


@then(parsers.parse('the response should contain error message "{message}"'))
def response_should_contain_error_message(context, message: str):
    """Assert that the response contains a specific error message."""
    response = context.get("response")
    assert response is not None, "No response found in context"
    data = response.json()
    
    # Check various error message locations
    error_message = data.get("message") or data.get("detail", {}).get("message")
    assert error_message is not None, f"No error message in response: {data}"
    assert message.lower() in error_message.lower(), (
        f"Expected error message containing '{message}', got '{error_message}'"
    )


# =============================================================================
# JSON Response Helpers
# =============================================================================

@then("the response should have a valid UUID id")
def response_should_have_valid_uuid(context):
    """Assert that the response contains a valid UUID in the 'id' field."""
    from uuid import UUID
    
    response = context.get("response")
    assert response is not None, "No response found in context"
    data = response.json()
    assert "id" in data, f"No 'id' field in response: {data}"
    
    try:
        UUID(data["id"])
    except (ValueError, TypeError) as e:
        pytest.fail(f"Invalid UUID in response: {data['id']} - {e}")


@then(parsers.parse('the response list should have {count:d} items'))
def response_list_should_have_count(context, count: int):
    """Assert that a list response contains a specific number of items."""
    response = context.get("response")
    assert response is not None, "No response found in context"
    data = response.json()
    
    # Handle various list response formats
    if isinstance(data, list):
        actual_count = len(data)
    elif "repositories" in data:
        actual_count = len(data["repositories"])
    elif "sessions" in data:
        actual_count = len(data["sessions"])
    elif "items" in data:
        actual_count = len(data["items"])
    else:
        pytest.fail(f"Could not find list in response: {data}")
    
    assert actual_count == count, (
        f"Expected {count} items, got {actual_count}"
    )


@then(parsers.parse('the response list should have at least {count:d} items'))
def response_list_should_have_at_least_count(context, count: int):
    """Assert that a list response contains at least a specific number of items."""
    response = context.get("response")
    assert response is not None, "No response found in context"
    data = response.json()
    
    # Handle various list response formats
    if isinstance(data, list):
        actual_count = len(data)
    elif "repositories" in data:
        actual_count = len(data["repositories"])
    elif "sessions" in data:
        actual_count = len(data["sessions"])
    elif "items" in data:
        actual_count = len(data["items"])
    else:
        pytest.fail(f"Could not find list in response: {data}")
    
    assert actual_count >= count, (
        f"Expected at least {count} items, got {actual_count}"
    )

