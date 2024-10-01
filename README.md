### **Pantasa Contribution Guide**

Welcome to the **Pantasa** project! This guide provides a detailed explanation of how to contribute to the Pantasa project, including steps for setting up the development environment, creating branches, making contributions, and submitting pull requests. Please follow these guidelines to ensure a smooth and collaborative workflow.

---

### **Table of Contents:**
1. [Project Overview](#project-overview)
2. [Getting Started](#getting-started)
3. [Branching Workflow](#branching-workflow)
4. [Making Contributions](#making-contributions)
5. [Handling Conflicts](#handling-conflicts)
6. [Pull Request Guidelines](#pull-request-guidelines)
7. [Testing Your Code](#testing-your-code)
8. [Commit Message Guidelines](#commit-message-guidelines)

---

### **Project Overview**

**Pantasa** is a Tagalog grammar correction system that combines rule-based error detection with deep learning-based iterative error correction. The goal of this project is to create an accurate and efficient grammar checker for the Tagalog language.

---

### **Getting Started**

#### Step 1: Clone the Repository

Start by cloning the Pantasa project repository to your local machine:

```bash
git clone https://github.com/BJCul/Pantasa.git
cd pantasa
```

#### Step 2: Set up a Virtual Environment

It is highly recommended to use a virtual environment to manage dependencies:

```bash
# Install virtualenv if you don't have it
pip install virtualenv

# Create a virtual environment in the project root
virtualenv pantasa_env

# Activate the environment (Windows)
pantasa_env\Scripts\activate

# Activate the environment (Linux/Mac)
source pantasa_env/bin/activate
```

#### Step 3: Install Dependencies

Install the required Python libraries listed in `requirements.txt`:

```bash
pip install -r requirements.txt
```

---

### **Branching Workflow**

We use a **Gitflow-inspired** branching strategy to manage the development workflow. This ensures that the `main` branch remains stable, while development happens in separate branches.

#### 1. **Main Branch** (`main`)

The `main` branch contains the latest stable version of the project. No direct commits are allowed in the `main` branch. All changes must come from feature branches through **pull requests**.

#### 2. **Feature Branches**

Create a new branch for each feature or bug fix. Always branch off from `main` and name your branch according to the feature or issue you're working on. Example branch names:
- `feature/error-detection`
- `feature/file-upload`
- `bugfix/preprocessing-tokenization`

#### Step 1: Pull Latest Changes from `main`

Before creating a new branch, ensure your local `main` branch is up-to-date with the remote repository.

```bash
git checkout main
git pull origin main
```

#### Step 2: Create a New Branch

Create a new branch for the feature you are working on:

```bash
git checkout -b feature/<feature-name>
```

Example:

```bash
git checkout -b feature/error-detection
```

#### Step 3: Push the Feature Branch to Remote

After making some commits, push the branch to the remote repository:

```bash
git push origin feature/<feature-name>
```

---

### **Making Contributions**

#### Step 1: Work on Your Feature

Make your changes to the codebase, following the project’s structure and conventions. Ensure that each unit of work is logically grouped into separate commits.

#### Step 2: Stage Your Changes

When you're ready to commit, stage the changes using:

```bash
git add <file-name>
```

Alternatively, to add all modified files:

```bash
git add .
```

#### Step 3: Commit Your Changes

Write descriptive and concise commit messages:

```bash
git commit -m "Implement hybrid n-gram error detection"
```

Follow the [Commit Message Guidelines](#commit-message-guidelines) to ensure consistency.

#### Step 4: Push Your Changes

Push your changes to the remote feature branch:

```bash
git push origin feature/<feature-name>
```

---

### **Handling Conflicts**

If your branch has conflicts with the `main` branch, follow these steps to resolve them:

#### Step 1: Switch to Your Branch

```bash
git checkout feature/<feature-name>
```

#### Step 2: Pull the Latest Changes from `main`

```bash
git pull origin main
```

#### Step 3: Resolve Conflicts

Git will alert you if there are any conflicts. Open the conflicting files, resolve the issues manually, and then add the resolved files:

```bash
git add <file-name>
```

#### Step 4: Commit the Merge

```bash
git commit -m "Resolve merge conflicts"
```

---

### **Pull Request Guidelines**

Once you’ve finished working on your feature, create a **pull request** (PR) to merge your feature branch into the `main` branch.

#### Step 1: Create a Pull Request

1. Go to the GitHub repository and click on the "Pull Requests" tab.
2. Click **New Pull Request**.
3. Select your feature branch as the **compare** branch and `main` as the **base** branch.

#### Step 2: Complete the Pull Request

1. Write a detailed description of the changes introduced by your branch.
2. Request reviews from other team members.
3. Ensure that your branch has no merge conflicts with `main`.
4. After approval, the branch will be merged into `main`.

---

### **Testing Your Code**

Before submitting a pull request, run tests to ensure that your changes do not break existing functionality.

#### Step 1: Run Unit Tests

Navigate to the `tests/` directory and run the unit tests:

```bash
python -m unittest discover tests
```

Ensure that all tests pass before submitting your pull request.

#### Step 2: Add New Tests

If your feature introduces new functionality, add tests to cover that feature in the `tests/` directory. For example:
- `tests/test_preprocessing.py`
- `tests/test_error_detection.py`

---

### **Commit Message Guidelines**

Please follow these commit message guidelines to keep the Git history clean and readable:

1. **Start with a verb**: Use a verb to describe the change. Example verbs: "Add", "Fix", "Implement", "Refactor", etc.
2. **Be concise and descriptive**: Example: `"Implement hybrid n-gram error detection"`
3. **Use imperative mood**: Example: `"Fix bug in preprocessing tokenization"`

---

### **Example Contribution Workflow**

Here’s a typical workflow example:

1. **Sync your local `main`**:
   ```bash
   git checkout main
   git pull origin main
   ```

2. **Create a new branch**:
   ```bash
   git checkout -b feature/error-detection
   ```

3. **Make changes and commit**:
   ```bash
   git add .
   git commit -m "Implement hybrid n-gram error detection"
   ```

4. **Push to the remote repository**:
   ```bash
   git push origin feature/error-detection
   ```

5. **Create a pull request** on GitHub:
   - Go to the "Pull Requests" tab, compare the branch with `main`, and create a PR.

6. **Address feedback** (if necessary), fix conflicts, and merge the PR after approval.

---

### Conclusion

Following this contribution guide will help ensure that we maintain a well-organized and clean codebase, allowing everyone to collaborate effectively. Don’t hesitate to ask for help or clarification if you encounter any issues!

Happy coding! 👩‍💻👨‍💻