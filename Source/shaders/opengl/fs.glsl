#version 330 core
in vec2 tex_c;

out vec4 FragColor;

uniform sampler2D aTexture;

float colormap_red(float x) {
    if (x < 0.7) {
        return 4.0 * x - 1.5;
    } else {
        return -4.0 * x + 4.5;
    }
}

float colormap_green(float x) {
    if (x < 0.5) {
        return 4.0 * x - 0.5;
    } else {
        return -4.0 * x + 3.5;
    }
}

float colormap_blue(float x) {
    if (x < 0.3) {
       return 4.0 * x + 0.5;
    } else {
       return -4.0 * x + 2.5;
    }
}

float colormap_grey(float x)
{
	return (x + 1.0) / 2.0;
}

void main()
{
    //FragColor = vec4(tex_c.x * tex_c.y * 1.0f, 0.0f, 0.0f, 1.0f);
    vec4 texture = texture(aTexture, tex_c);
    FragColor = vec4(colormap_grey(texture.x), colormap_grey(texture.x), colormap_grey(texture.x), 1.0);
}