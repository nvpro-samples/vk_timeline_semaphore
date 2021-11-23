// Copyright 2021 NVIDIA CORPORATION
// SPDX-License-Identifier: Apache-2.0
#pragma once

#include <string>
#include <vector>

#include "nvpsystem.hpp"

static const std::string installPath  = NVPSystem::exePath() + PROJECT_NAME "/";
static const std::string repoRootPath = NVPSystem::exePath() + PROJECT_RELDIRECTORY "/";

// This is used by nvh::findFile to search for shader files.
static const std::vector<std::string> searchPaths = {
    installPath,
    repoRootPath,
};
