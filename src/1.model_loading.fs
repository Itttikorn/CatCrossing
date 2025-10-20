#version 330 core
out vec4 FragColor;

in VS_OUT {
    vec2 TexCoords;
    vec3 FragPos;
    vec3 Normal;
} fs_in;

uniform sampler2D texture_diffuse1; 
uniform int  useTexture;             
uniform vec3 materialColor;          

uniform vec3 lightPos;               
uniform vec3 viewPos;                
uniform vec3 lightColor;

void main()
{
    // Base color: use real texture if present, otherwise MTL color.
    vec3 base = (useTexture == 1)
              ? texture(texture_diffuse1, fs_in.TexCoords).rgb
              : materialColor;

    
    vec3 N = normalize(fs_in.Normal);
    vec3 V = normalize(viewPos - fs_in.FragPos);

    // Main light
    vec3 L  = normalize(lightPos - fs_in.FragPos);
    float diff = max(dot(N, L), 0.0);
    float spec = pow(max(dot(normalize(L + V), N), 0.0), 32.0);

    // Headlight 
    vec3 Lh = normalize(viewPos - fs_in.FragPos);
    float diffH = max(dot(N, Lh), 0.0);
    float specH = pow(max(dot(normalize(Lh + V), N), 0.0), 32.0);

    vec3 ambient  = 0.45 * base;
    vec3 diffuse  = (0.75*diff + 0.45*diffH) * base * lightColor;
    vec3 specular = (0.25*spec + 0.15*specH) * lightColor;

    vec3 color = ambient + diffuse + specular;
    color = pow(color, vec3(1.0/2.2));
    FragColor = vec4(color, 1.0);
}
