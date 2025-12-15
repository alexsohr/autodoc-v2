Feature: Wiki Generation
  As an API user
  I want to generate and access documentation wikis
  So that I can understand and share knowledge about repositories

  Background:
    Given I am authenticated as an admin user
    And I have an analyzed repository

  # ==========================================================================
  # Wiki Structure
  # ==========================================================================

  Scenario: Get wiki structure for a repository
    When I get the wiki structure
    Then the response status should be 200
    And the wiki structure should contain pages
    And the wiki structure should contain sections
    And the wiki structure should contain root sections

  Scenario: Get wiki structure with content included
    When I get the wiki structure with content included
    Then the response status should be 200
    And the wiki pages should include content

  Scenario: Get wiki structure filtered by section
    When I get the wiki structure for section "introduction"
    Then the response status should be 200

  Scenario: Fail to get wiki for non-existent repository
    When I get wiki for repository "00000000-0000-0000-0000-000000000000"
    Then the response status should be 404

  # ==========================================================================
  # Wiki Pages
  # ==========================================================================

  Scenario: Get a specific wiki page
    When I get wiki page "overview"
    Then the response status should be 200
    And the response should contain "id"
    And the response should contain "title"

  Scenario: Get wiki page in JSON format
    When I get wiki page "overview" in format "json"
    Then the response status should be 200
    And the response should contain "content"

  Scenario: Get wiki page in Markdown format
    When I get wiki page "overview" in format "markdown"
    Then the response status should be 200
    And the response content type should be "text/markdown"

  Scenario: Fail to get non-existent wiki page
    When I get wiki page "non-existent-page"
    Then the response status should be 404

  # ==========================================================================
  # Wiki Page Content Quality
  # ==========================================================================

  Scenario: Wiki page has proper structure
    When I get wiki page "overview"
    Then the response status should be 200
    And the page should have a title
    And the page should have a description
    And the page should have an importance level

  Scenario: Wiki page includes related pages
    When I get wiki page "overview"
    Then the response status should be 200
    And the page should have related pages

  Scenario: Wiki page references source files
    When I get wiki page "overview"
    Then the response status should be 200
    And the page should have file paths

  # ==========================================================================
  # Repository Files
  # ==========================================================================

  Scenario: Get repository file list
    When I get the repository files
    Then the response status should be 200
    And the response should contain "files"
    And the response should contain "total"

  Scenario: Get repository files filtered by language
    When I get the repository files with language filter "python"
    Then the response status should be 200
    And the response should contain "files"

  Scenario: Get repository files with path pattern
    When I get the repository files with path pattern "src/"
    Then the response status should be 200

  Scenario: Get repository files with pagination
    When I get the repository files with limit 10 and offset 0
    Then the response status should be 200

  # ==========================================================================
  # Documentation Pull Request
  # ==========================================================================

  Scenario: Create documentation pull request
    When I create a documentation pull request
    Then the response status should be 201
    And the response should contain "pull_request_url"
    And the response should contain "branch_name"
    And the response should contain "files_changed"
    And the response should contain "commit_sha"

  Scenario: Create documentation PR with custom title
    When I create a documentation pull request with title "Update docs"
    Then the response status should be 201
    And the response should contain "pull_request_url"

  Scenario: Create documentation PR with target branch
    When I create a documentation pull request to branch "develop"
    Then the response status should be 201

  Scenario: PR URL should be a valid GitHub URL
    When I create a documentation pull request
    Then the response status should be 201
    And the pull request URL should be a valid GitHub URL

  Scenario: PR branch follows naming convention
    When I create a documentation pull request
    Then the response status should be 201
    And the branch name should contain "autodoc"

  Scenario: PR includes documentation file changes
    When I create a documentation pull request
    Then the response status should be 201
    And the files changed should be documentation files

