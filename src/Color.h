#include <list>

class Color
{
    public:
        Color()
        {
            r_=1.0;
            g_=1.0;
            b_=1.0;
        };

        Color(float r, float g, float b) : r_(r), g_(g), b_(b) {};

        Color(const Color &c)
        {
            r_=c.r_;
            g_=c.g_;
            b_=c.b_;
        }

        float r_;
        float g_;
        float b_;
};

class ColorPicker
{
    public:

        ColorPicker(int nColors) : nColors_(nColors)
    {
        if(nColors < 2)
            return;
        
        float dx = 1.0f / (float) (nColors - 1);
        for (int i = 0; i < nColors_; i++) {
            Color newcolor = setColor(i*dx);
            vColors_.push_back(newcolor);
        }
    };

        Color setColor(float x)
        {
            Color c;

            float r = 0.0f;
            float g = 0.0f;
            float b = 1.0f;
            if (x >= 0.0f && x < 0.2f) {
                x = x / 0.2f;
                r = 0.0f;
                g = x;
                b = 1.0f;
            } else if (x >= 0.2f && x < 0.4f) {
                x = (x - 0.2f) / 0.2f;
                r = 0.0f;
                g = 1.0f;
                b = 1.0f - x;
            } else if (x >= 0.4f && x < 0.6f) {
                x = (x - 0.4f) / 0.2f;
                r = x;
                g = 1.0f;
                b = 0.0f;
            } else if (x >= 0.6f && x < 0.8f) {
                x = (x - 0.6f) / 0.2f;
                r = 1.0f;
                g = 1.0f - x;
                b = 0.0f;
            } else if (x >= 0.8f && x <= 1.0f) {
                x = (x - 0.8f) / 0.2f;
                r = 1.0f;
                g = 0.0f;
                b = x;
            }
            c.r_ = 255*r;
            c.g_ = 255*g;
            c.b_ = 255*b;
            
            return c;
        }

        Color getColor(int a)
        {
            if(a < vColors_.size())
                return vColors_.at(a);
            else
            {
                Color default_color;
                return default_color; 
            }
        }

    private:
        int nColors_;
        std::vector<Color> vColors_;
};


