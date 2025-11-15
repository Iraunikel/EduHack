<!-- 44d86b71-4c44-4df3-86c3-bde29a4da9ad cb3dc844-2c69-43d0-92f3-b5aade4ea200 -->
# Replace langdetect with Polyglot for Luxembourgish Support

## Current Situation

- Using `langdetect==1.0.9` which supports ~55 languages
- Luxembourgish (lb) is NOT supported by langdetect
- Project requires detection of fr, de, en, lb languages
- Test data already includes Luxembourgish content
- Current implementation uses minimum 10 characters for detection

## Solution: Polyglot

**Why Polyglot:**

- Supports 130+ languages including Luxembourgish (lb) ✓
- Uses Google's CLD2 (Compact Language Detector 2) via pycld2
- Actively maintained open-source library
- Well-documented at https://polyglot.readthedocs.io/
- Returns ISO 639-1 language codes (compatible with current usage)
- Good accuracy for multilingual content

**Important considerations:**

- Requires downloading language models: `polyglot download LANG:lb`
- Has additional dependencies: polyglot, pycld2, pyicu (ICU), morfessor, numpy
- Different API than langdetect: uses `Detector(text).language.code`
- May require system-level ICU library installation

## Implementation Plan

### 1. Update dependencies

- Remove `langdetect==1.0.9` from `requirements.txt`
- Add `polyglot` and `pycld2` packages
- Note: pyicu and morfessor are auto-installed as polyglot dependencies
- Document language model download requirement in README/CHANGELOG

### 2. Update detector.py

- Replace `langdetect` import with `polyglot.detect.Detector`
- Update `detect_language()` function to use Polyglot API
- Maintain same function signature: `detect_language(text: str) -> str`
- Handle exceptions: catch `UnknownLanguage` and other Polyglot exceptions
- Ensure minimum text length handling (keep 10 character minimum)
- Extract language code from `detector.language.code`

### 3. Language code mapping

- Verify language codes: fr, de, en, lb are correctly returned
- Polyglot returns ISO 639-1 codes (same as langdetect)
- Ensure "unknown" fallback behavior is maintained
- Handle cases where language detection fails or returns None

### 4. Setup and installation

- Create setup instructions for language model download
- Document: `polyglot download LANG:lb` command
- Consider adding model download to setup script or CLI
- Handle missing language models gracefully with helpful error messages

### 5. Testing

- Test with existing test data (fr, de, en, lb)
- Verify Luxembourgish detection works correctly
- Ensure backward compatibility with other languages (fr, de, en)
- Test error handling for short/invalid text
- Test with missing language models

### 6. Documentation

- Update CHANGELOG.md with dependency change
- Update ARCHITECTURE.md to reflect Polyglot usage
- Update README.md with installation instructions for language models
- Update any comments in code

## Files to Modify

- `requirements.txt`: Update dependencies (remove langdetect, add polyglot pycld2)
- `agents/detector.py`: Update language detection implementation
- `CHANGELOG.md`: Document the change and setup requirements
- `README.md`: Add language model download instructions
- `ARCHITECTURE.md`: Update to reflect Polyglot usage

## Risk Assessment

- **Medium risk**: Different API and additional setup requirements
- **Dependencies**: Requires system ICU library (may need system package installation)
- **Setup**: Requires manual language model download step
- **Compatibility**: API changes needed but function signature stays the same
- **Testing**: Will verify with existing test data before committing

## Additional Notes

- Polyglot requires language models to be downloaded separately
- ICU library may need to be installed at system level (e.g., `brew install icu4c` on macOS)
- Consider adding language model download to installation/setup process
- May want to add helper function to check if language models are available

### To-dos

- [ ] Update requirements.txt: remove langdetect, add lingua-language-detector
- [ ] Update detector.py: replace langdetect import with lingua import
- [ ] Update detect_language() function to use Lingua API while maintaining same signature
- [ ] Test language detection with all languages (fr, de, en, lb) using test data
- [ ] Update CHANGELOG.md to document the dependency change