#!/bin/bash -x
set -e

readonly REGION="westus2"
readonly CLUSTER_NAME="onnxr"

readonly containerUrl=https://anybuild${CLUSTER_NAME}${REGION}.blob.core.windows.net/clientreleases
readonly ANYBUILD_HOME="$AGENT_WORKFOLDER/Microsoft/AnyBuild"
readonly channel="Dogfood"
# Download a file from internet
function GetFile {
  local uri=$1
  local path=$2
  local force=${3:-false}
  local download_retries=${4:-5}
  local retry_wait_time_seconds=${5:-30}

  if [[ -f $path ]]; then
    if [[ $force = false ]]; then
      echo "File '$path' already exists. Skipping download"
      return 0
    else
      rm -rf $path
    fi
  fi

  if [[ -f $uri ]]; then
    echo "'$uri' is a file path, copying file to '$path'"
    cp $uri $path
    return $?
  fi

  echo "Downloading $uri"
  # Use aria2c if available, otherwise use curl
  if command -v aria2c > /dev/null; then
    aria2c -q -d $(dirname $path) -o $(basename $path) "$uri"
  else
    curl "$uri" -sSL --retry $download_retries --retry-delay $retry_wait_time_seconds --create-dirs -o "$path" --fail
  fi

  return $?
}

if [[ ! -f "$ANYBUILD_HOME/AnyBuild.sh" ]]; then
    echo
    echo "=== Installing AnyBuild client ==="
    echo

	anyBuildClientBaseDir=$AGENT_WORKFOLDER/AnyBuild
	mkdir -p $anyBuildClientBaseDir

	echo "Downloading and running AnyBuildUpdater from $channel channel from $containerUrl"
	GetFile $containerUrl/ReleasesLinux.json $anyBuildClientBaseDir/ReleasesLinux.json

	currentRelease=$(cat $anyBuildClientBaseDir/ReleasesLinux.json | python -c "import sys, json; print(json.load(sys.stdin)['${channel}Channel']['Release'])")

	updaterDir=$anyBuildClientBaseDir/BootstrapUpdater_$currentRelease
	if [ -d $updaterDir ]; then
	  echo "Deleting $updaterDir"
	  rm -r $updaterDir
	fi

	updaterArchive="$updaterDir/AnyBuildUpdater.tar.gz"
	mkdir -p $updaterDir
	GetFile $containerUrl/$currentRelease/Linux/AnyBuildUpdater.tar.gz $updaterArchive
	tar xzf $updaterArchive --directory $updaterDir

	updaterBinary=$updaterDir/AnyBuildUpdater

	# Ensure the permission to execute is set because if AnyBuild validation/release workflow creates the resulting archive on Windows machine which doesn't preserve this flag.
	chmod +x $updaterBinary
	cd $AGENT_WORKFOLDER
	echo "Executing: $updaterBinary --Channel $channel --ReleaseContainerUrl $containerUrl"
	$updaterBinary --Channel $channel --ReleaseContainerUrl $containerUrl
fi