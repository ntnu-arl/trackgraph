#pragma once

#include <memory>
#include <string>

#include "hydra/openset/embedding_group.h"

namespace hydra {

struct OpenVocabPromptBank : public EmbeddingGroup {
  using Ptr = std::shared_ptr<OpenVocabPromptBank>;

  std::string encoder_id;
};

}  // namespace hydra
