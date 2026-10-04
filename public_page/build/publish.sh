#!/usr/bin/env bash
# Publish public_page/site/ as the branch gh-pages: the branch holds the content of site/ only.
# The working tree is not touched; the commit is built from the tree object of HEAD.
#
#   bash public_page/build/publish.sh            # update the local branch gh-pages
#   bash public_page/build/publish.sh --push     # ... and push it to origin
#
# Run it on main, with public_page/ committed. It refuses to publish when
#   - the current branch is not main,
#   - public_page/ has uncommitted changes,
#   - the full build (figures, claims, every check) does not end OK,
#   - the build changed a file: the committed chart data, manifest or checklist was not a fresh build.
# PUBLISH_MESSAGE_SUFFIX, if set, is appended to the commit message (for example trailers).
set -euo pipefail
cd "$(git rev-parse --show-toplevel)"
PAGE=public_page
SITE=$PAGE/site

branch=$(git rev-parse --abbrev-ref HEAD)
if [ "$branch" != main ]; then
    echo "publish: the current branch is '$branch', not main; nothing was published" >&2
    exit 1
fi

if [ -n "$(git status --porcelain -- "$PAGE")" ]; then
    echo "publish: $PAGE has uncommitted changes; build, check and commit first" >&2
    exit 1
fi

# The whole build, not only the checks: the chart data file is generated, so the published copy must be
# the one a fresh build gives. The build writes no time stamp, so a fresh build of committed files changes nothing.
if ! PYTHONDONTWRITEBYTECODE=1 python $PAGE/build/build_public.py --quiet; then
    echo "publish: the build or a check fails, nothing was published (see $PAGE/exposure_checklist.md)" >&2
    exit 1
fi
if [ -n "$(git status --porcelain -- "$PAGE")" ]; then
    echo "publish: the build changed files under $PAGE (listed below): the committed copy was not a fresh build." >&2
    echo "         Look at the difference, commit it, then publish again. Nothing was published." >&2
    git status --porcelain -- "$PAGE" >&2
    exit 1
fi

tree=$(git rev-parse "HEAD:$SITE")
head=$(git rev-parse --short HEAD)
# The parent is the last published commit: the local branch, or the branch on origin in a checkout that
# has never published (a fresh clone, another machine).
# (the explicit refspec also works in a clone that tracks main only)
git fetch -q origin "+refs/heads/gh-pages:refs/remotes/origin/gh-pages" 2>/dev/null || true
local_tip=$(git rev-parse -q --verify refs/heads/gh-pages || true)
remote_tip=$(git rev-parse -q --verify refs/remotes/origin/gh-pages || true)
parent=$local_tip
if [ -n "$remote_tip" ]; then
    # origin is at or ahead of the local branch (or there is no local branch): continue from origin
    if [ -z "$local_tip" ] || git merge-base --is-ancestor "$local_tip" "$remote_tip"; then
        parent=$remote_tip
    fi
fi
if [ -n "$parent" ] && [ "$(git rev-parse "$parent^{tree}")" = "$tree" ]; then
    git update-ref refs/heads/gh-pages "$parent"
    echo "publish: gh-pages already holds this version of the site"
else
    message="Publish the public page (main $head)"
    if [ -n "${PUBLISH_MESSAGE_SUFFIX:-}" ]; then
        message="$message"$'\n\n'"$PUBLISH_MESSAGE_SUFFIX"
    fi
    commit=$(git commit-tree "$tree" ${parent:+-p "$parent"} -m "$message")
    git update-ref refs/heads/gh-pages "$commit"
    echo "publish: gh-pages -> $(git rev-parse --short "$commit") (site of main $head)"
fi

if [ "${1:-}" = "--push" ]; then
    git push origin gh-pages
fi
