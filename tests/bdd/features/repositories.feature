Feature: Repository Management
  As an API user
  I want to manage code repositories
  So that I can generate documentation for my codebases

  Background:
    Given I am authenticated as an admin user

  # ==========================================================================
  # Repository Registration
  # ==========================================================================

  Scenario: Register a new GitHub repository
    When I register a repository with URL "https://github.com/yusufocaliskan/python-flask-mvc"
    Then the response status should be 201
    And the response should have a valid UUID id
    And the repository should have status "pending"
    And the repository provider should be "github"

  Scenario: Register a GitHub repository with custom branch
    When I register a repository with URL "https://github.com/yusufocaliskan/python-flask-mvc" and branch "main"
    Then the response status should be 201
    And the repository default branch should be "main"

  Scenario: Register a repository with auto-detected provider
    When I register a repository with URL "https://github.com/yusufocaliskan/python-flask-mvc"
    Then the response status should be 201
    And the repository provider should be "github"
    And the repository org should be "yusufocaliskan"
    And the repository name should be "python-flask-mvc"

  Scenario: Fail to register duplicate repository
    Given I have a registered repository with URL "https://github.com/yusufocaliskan/python-flask-mvc"
    When I register a repository with URL "https://github.com/yusufocaliskan/python-flask-mvc"
    Then the response status should be 409
    And the response should contain error message "already exists"

  Scenario: Fail to register repository with invalid URL
    When I register a repository with URL "not-a-valid-url"
    Then the response status should be 400

  Scenario: Fail to register repository with missing repo path
    When I register a repository with URL "https://github.com/"
    Then the response status should be 400

  # ==========================================================================
  # Repository Listing
  # ==========================================================================

  Scenario: List all repositories
    Given I have a registered repository with URL "https://github.com/yusufocaliskan/python-flask-mvc"
    When I list all repositories
    Then the response status should be 200
    And the response list should have at least 1 items
    And the response should contain "total"

  Scenario: List repositories with status filter
    Given I have a registered repository with URL "https://github.com/yusufocaliskan/python-flask-mvc"
    When I list repositories with status filter "pending"
    Then the response status should be 200
    And all repositories in the response should have status "pending"

  Scenario: List repositories with pagination
    Given I have a registered repository with URL "https://github.com/yusufocaliskan/python-flask-mvc"
    When I list repositories with limit 1 and offset 0
    Then the response status should be 200
    And the response list should have 1 items

  # ==========================================================================
  # Repository Details
  # ==========================================================================

  Scenario: Get repository details
    Given I have a registered repository with URL "https://github.com/yusufocaliskan/python-flask-mvc"
    When I get the repository details
    Then the response status should be 200
    And the response should contain "id"
    And the response should contain "url"
    And the response should contain "provider"
    And the response should contain "analysis_status"

  Scenario: Fail to get non-existent repository
    When I get repository with ID "00000000-0000-0000-0000-000000000000"
    Then the response status should be 404

  # ==========================================================================
  # Repository Analysis
  # ==========================================================================

  Scenario: Trigger repository analysis
    Given I have a registered repository with URL "https://github.com/yusufocaliskan/python-flask-mvc"
    When I trigger analysis for the repository
    Then the response status should be 202
    And the analysis response status should be "processing"
    And the response should contain "progress"

  Scenario: Trigger forced re-analysis
    Given I have a registered repository with URL "https://github.com/yusufocaliskan/python-flask-mvc"
    When I trigger forced analysis for the repository
    Then the response status should be 202

  Scenario: Get analysis status
    Given I have a registered repository with URL "https://github.com/yusufocaliskan/python-flask-mvc"
    When I get the analysis status for the repository
    Then the response status should be 200
    And the response should contain "status"
    And the response should contain "progress"

  # ==========================================================================
  # Repository Deletion
  # ==========================================================================

  Scenario: Delete a repository
    Given I have a registered repository with URL "https://github.com/yusufocaliskan/python-flask-mvc"
    When I delete the repository
    Then the response status should be 204
    And the repository should no longer exist

  Scenario: Fail to delete non-existent repository
    When I delete repository with ID "00000000-0000-0000-0000-000000000000"
    Then the response status should be 404

  # ==========================================================================
  # Webhook Configuration
  # ==========================================================================

  Scenario: Configure repository webhook
    Given I have a registered repository with URL "https://github.com/yusufocaliskan/python-flask-mvc"
    When I configure the webhook with secret "my-webhook-secret"
    Then the response status should be 200
    And the response should contain "webhook_configured"
    And the response should contain "setup_instructions"

  Scenario: Get webhook configuration
    Given I have a registered repository with URL "https://github.com/yusufocaliskan/python-flask-mvc"
    When I get the webhook configuration
    Then the response status should be 200
    And the response should contain "webhook_configured"
