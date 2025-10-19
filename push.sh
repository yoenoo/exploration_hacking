set -xe

# Read current version
VERSION=$(cat VERSION)

# Increment patch version (0.0.8 -> 0.0.9)
IFS='.' read -r major minor patch <<< "$VERSION"
patch=$((patch + 1))
NEW_VERSION="$major.$minor.$patch"

echo "Building version v$NEW_VERSION (previous: v$VERSION)"

# Build and push
docker build -t yoenoo/runpod-serverless-test:v$NEW_VERSION .
docker push yoenoo/runpod-serverless-test:v$NEW_VERSION

# Update version file
echo "$NEW_VERSION" > VERSION

echo "Successfully built and pushed v$NEW_VERSION"