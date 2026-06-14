# GitHub upload guide for anonymous review

## Important anonymity point

GitHub does not provide an ordinary "public but unlisted by secret link" mode.
A public repository can be viewed by anyone who can visit GitHub, and the owner
account is visible. A private repository is not accessible by link unless access
is granted. For double-blind review, use a neutral account that contains no name,
photo, institutional email, biography, website, or identifying commit metadata.

## Suggested workflow

1. Create a new GitHub account with a neutral username, for example
   `ce-abm-review` plus random numbers.
2. Use a neutral email address that does not contain your name or institution.
3. Do not add profile photo, location, institution, website, ORCID, personal
   email, or biography.
4. Create a new repository with a neutral name, for example
   `ce-abm-amsterdam-review`.
5. Set repository visibility to **Public** if reviewers need to open it only by
   link without being invited.
6. Upload the files in this package.
7. Check the repository page in a private/incognito browser window.
8. Copy the repository URL and use that URL in the manuscript code-availability
   statement.
9. After review/acceptance, transfer or mirror the repository to the author
   account, replace anonymous metadata, select the final license, and create a
   tagged release.

## Local command-line upload

```bash
git init
git add .
git commit -m "Initial anonymous review version"
git branch -M main
git remote add origin https://github.com/ANON_ACCOUNT/ce-abm-amsterdam-review.git
git push -u origin main
```

## Metadata checks before upload

Run these searches locally before the first commit:

```bash
grep -Rni "your-name\|your-email\|university\|orcid\|OneDrive\|Desktop\|C:/Users" .
git config user.name
git config user.email
```

For anonymous commits, set local repository-only Git identity:

```bash
git config user.name "Anonymous Authors"
git config user.email "anonymous@example.com"
```
