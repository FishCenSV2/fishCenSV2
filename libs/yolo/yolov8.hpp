#include <vector>
#include <string>

//Need to figure out how this class should be set up.

class YoloV8 {
    private:
        std::vector<char> _engine_data;
    public:
        YoloV8();
        void readEngineFile(const std::string& engineFileName, std::vector<char>& engineData);
};