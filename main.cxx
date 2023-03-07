/*
 * //Make a cube object to render
 * Keyboard To move an object
 * Make Scene
 * Add collision
 * Gravity
 * Jumping
 * Texturing
 * Better Lighting
 * Game mechanics
 * Better player model
 */

#include <iostream>
#include <stdio.h>
#include <stdlib.h>
#include <vector>
#include <unistd.h>

using std::endl;
using std::cerr;

#define GL_SILENCE_DEPRECATION

#include <GLFW/glfw3.h>

#define GLM_ENABLE_EXPERIMENTAL
#include <glm/vec3.hpp>
#include <glm/vec4.hpp>
#include <glm/mat4x4.hpp>
#include <glm/gtx/string_cast.hpp>
#include <glm/gtc/type_ptr.hpp>
#include <glm/gtc/matrix_transform.hpp> 

#ifdef __APPLE__
#include <OpenGL/gl3.h>
#include <OpenGL/gl3ext.h>
#endif

class RenderManager;

void KeyCallback(GLFWwindow *window, int key, int scancode, int action, int mods);

void SetUpLevel(RenderManager &);
void SetUpCharacter(RenderManager &);

const char *GetVertexShader();
const char *GetFragmentShader();

class Triangle
{
  public:
    glm::vec3 v0;
    glm::vec3 v1;
    glm::vec3 v2;
};

std::vector<Triangle> SplitTriangle(std::vector<Triangle> &list)
{
    std::vector<Triangle> output(4*list.size());
    output.resize(4*list.size());
    for (unsigned int i = 0 ; i < list.size() ; i++)
    {
        Triangle t = list[i];
        glm::vec3 vmid1, vmid2, vmid3;
        vmid1 = (t.v0 + t.v1) / 2.0f;
        vmid2 = (t.v1 + t.v2) / 2.0f;
        vmid3 = (t.v0 + t.v2) / 2.0f;
        output[4*i+0].v0 = t.v0;
        output[4*i+0].v1 = vmid1;
        output[4*i+0].v2 = vmid3;
        output[4*i+1].v0 = t.v1;
        output[4*i+1].v1 = vmid2;
        output[4*i+1].v2 = vmid1;
        output[4*i+2].v0 = t.v2;
        output[4*i+2].v1 = vmid3;
        output[4*i+2].v2 = vmid2;
        output[4*i+3].v0 = vmid1;
        output[4*i+3].v1 = vmid2;
        output[4*i+3].v2 = vmid3;
    }
    return output;
}

void 
PushVertex(std::vector<float>& coords, const glm::vec3& v)
{
  coords.push_back(v.x);
  coords.push_back(v.y);
  coords.push_back(v.z);
}

// Sets up a cylinder that is the circle x^2+y^2=1 extruded from
// Z=0 to Z=1.
void 
GetCylinderData(std::vector<float>& coords, std::vector<float>& normals)
{
  int nfacets = 30;
  for (int i = 0 ; i < nfacets ; i++)
  {
    double angle = 3.14159*2.0*i/nfacets;
    double nextAngle = (i == nfacets-1 ? 0 : 3.14159*2.0*(i+1)/nfacets);
    glm::vec3 fnormal(0.0f, 0.0f, 1.0f);
    glm::vec3 bnormal(0.0f, 0.0f, -1.0f);
    glm::vec3 fv0(0.0f, 0.0f, 1.0f);
    glm::vec3 fv1(cos(angle), sin(angle), 1);
    glm::vec3 fv2(cos(nextAngle), sin(nextAngle), 1);
    glm::vec3 bv0(0.0f, 0.0f, 0.0f);
    glm::vec3 bv1(cos(angle), sin(angle), 0);
    glm::vec3 bv2(cos(nextAngle), sin(nextAngle), 0);
    // top and bottom circle vertices
    PushVertex(coords, fv0);
    PushVertex(normals, fnormal);
    PushVertex(coords, fv1);
    PushVertex(normals, fnormal);
    PushVertex(coords, fv2);
    PushVertex(normals, fnormal);
    PushVertex(coords, bv0);
    PushVertex(normals, bnormal);
    PushVertex(coords, bv1);
    PushVertex(normals, bnormal);
    PushVertex(coords, bv2);
    PushVertex(normals, bnormal);
    // curves surface vertices
    glm::vec3 v1normal(cos(angle), sin(angle), 0);
    glm::vec3 v2normal(cos(nextAngle), sin(nextAngle), 0);
    //fv1 fv2 bv1
    PushVertex(coords, fv1);
    PushVertex(normals, v1normal);
    PushVertex(coords, fv2);
    PushVertex(normals, v2normal);
    PushVertex(coords, bv1);
    PushVertex(normals, v1normal);
    //fv2 bv1 bv2
    PushVertex(coords, fv2);
    PushVertex(normals, v2normal);
    PushVertex(coords, bv1);
    PushVertex(normals, v1normal);
    PushVertex(coords, bv2);
    PushVertex(normals, v2normal);
  }
}

// Sets up a sphere with equation x^2+y^2+z^2=1
void
GetSphereData(std::vector<float>& coords, std::vector<float>& normals)
{
  int recursionLevel = 3;
  std::vector<Triangle> list;
  {
    Triangle t;
    t.v0 = glm::vec3(1.0f,0.0f,0.0f);
    t.v1 = glm::vec3(0.0f,1.0f,0.0f);
    t.v2 = glm::vec3(0.0f,0.0f,1.0f);
    list.push_back(t);
  }
  for (int r = 0 ; r < recursionLevel ; r++)
  {
      list = SplitTriangle(list);
  }

  for (int octant = 0 ; octant < 8 ; octant++)
  {
    glm::mat4 view(1.0f);
    float angle = 90.0f*(octant%4);
    if(angle != 0.0f)
      view = glm::rotate(view, glm::radians(angle), glm::vec3(1, 0, 0));
    if (octant >= 4)
      view = glm::rotate(view, glm::radians(180.0f), glm::vec3(0, 0, 1));
    for(int i = 0; i < list.size(); i++)
    {
      Triangle t = list[i];
      float mag_reci;
      glm::vec3 v0 = view*glm::vec4(t.v0, 1.0f);
      glm::vec3 v1 = view*glm::vec4(t.v1, 1.0f);
      glm::vec3 v2 = view*glm::vec4(t.v2, 1.0f);
      mag_reci = 1.0f / glm::length(v0);
      v0 = glm::vec3(v0.x * mag_reci, v0.y * mag_reci, v0.z * mag_reci);
      mag_reci = 1.0f / glm::length(v1);
      v1 = glm::vec3(v1.x * mag_reci, v1.y * mag_reci, v1.z * mag_reci);
      mag_reci = 1.0f / glm::length(v2);
      v2 = glm::vec3(v2.x * mag_reci, v2.y * mag_reci, v2.z * mag_reci);
      PushVertex(coords, v0);
      PushVertex(coords, v1);
      PushVertex(coords, v2);
      PushVertex(normals, v0);
      PushVertex(normals, v1);
      PushVertex(normals, v2);
    }
  }
}

void 
GetCubeData(std::vector<float>& coords, std::vector<float>& normals)
{
  std::vector<Triangle> list;
  Triangle t0, t1;
  
  glm::vec3 p0 = glm::vec3(1.0f, 1.0f, 1.0f);
  glm::vec3 p1 = glm::vec3(1.0f, -1.0f, 1.0f);
  glm::vec3 p2 = glm::vec3(-1.0f, -1.0f, 1.0f);
  glm::vec3 p3 = glm::vec3(-1.0f, 1.0f, 1.0f);
  glm::vec3 p4 = glm::vec3(-1.0f, -1.0f, -1.0f);
  glm::vec3 p5 = glm::vec3(1.0f, -1.0f, -1.0f);
  glm::vec3 p6 = glm::vec3(1.0f, 1.0f, -1.0f);
  glm::vec3 p7 = glm::vec3(-1.0f, 1.0f, -1.0f);

  // Front Face z = 1
  t0.v0 = p0;
  t0.v1 = p1;
  t0.v2 = p2;
  list.push_back(t0);
  t1.v0 = p0;
  t1.v1 = p2;
  t1.v2 = p3;
  list.push_back(t1); 
  
  // Back Face z = -1
  t0.v0 = p6;
  t0.v1 = p5;
  t0.v2 = p4;
  list.push_back(t0);
  t1.v0 = p6;
  t1.v1 = p4;
  t1.v2 = p7;
  list.push_back(t1); 
  
  // Top Face y = 1
  t0.v0 = p0;
  t0.v1 = p3;
  t0.v2 = p7;
  list.push_back(t0);
  t1.v0 = p0;
  t1.v1 = p6;
  t1.v2 = p7;
  list.push_back(t1); 
 
  // Bottom Face y = -1
  t0.v0 = p1;
  t0.v1 = p2;
  t0.v2 = p5;
  list.push_back(t0);
  t1.v0 = p2;
  t1.v1 = p4;
  t1.v2 = p5;
  list.push_back(t1); 
  
  // Left Face x = -1
  t0.v0 = p2;
  t0.v1 = p3;
  t0.v2 = p4;
  list.push_back(t0);
  t1.v0 = p3;
  t1.v1 = p4;
  t1.v2 = p7;
  list.push_back(t1); 
  
  // Right Face x = 1
  t0.v0 = p0;
  t0.v1 = p1;
  t0.v2 = p5;
  list.push_back(t0);
  t1.v0 = p0;
  t1.v1 = p5;
  t1.v2 = p6;
  list.push_back(t1); 

  for (int i = 0; i < list.size(); i++)
  {
    PushVertex(coords, list[i].v0);
    PushVertex(coords, list[i].v1);
    PushVertex(coords, list[i].v2);
    PushVertex(normals, list[i].v0);
    PushVertex(normals, list[i].v1);
    PushVertex(normals, list[i].v2);
  }
}

void 
_print_shader_info_log(GLuint shader_index) 
{
  int max_length = 2048;
  int actual_length = 0;
  char shader_log[2048];
  glGetShaderInfoLog(shader_index, max_length, &actual_length, shader_log);
  printf("shader info log for GL index %u:\n%s\n", shader_index, shader_log);
}

class RenderManager
{
  public:
   enum ShapeType
   {
      SPHERE,
      CYLINDER,
      CUBE
   };

                 RenderManager();
   void          SetView(glm::vec3 &c, glm::vec3 &, glm::vec3 &);
   void          SetUpGeometry();
   void          SetColor(double r, double g, double b);
   void          Render(ShapeType, glm::mat4 model);
   GLFWwindow   *GetWindow() { return window; };

  private:
   glm::vec3 color;
   GLuint sphereVAO;
   GLuint sphereNumPrimitives;
   GLuint cylinderVAO;
   GLuint cylinderNumPrimitives;
   GLuint cubeVAO;
   GLuint cubeNumPrimitives;  
   GLuint mloc;
   GLuint mvploc;
   GLuint colorloc;
   GLuint camloc;
   GLuint ldirloc;
   glm::mat4 projection;
   glm::mat4 view;
   GLuint shaderProgram;
   GLFWwindow *window;

   void SetUpWindowAndShaders();
   void MakeModelView(glm::mat4 &);
};

RenderManager::RenderManager()
{
  SetUpWindowAndShaders();
  SetUpGeometry();
  projection = glm::perspective(
        glm::radians(45.0f), (float)1000 / (float)1000,  5.0f, 100.0f);

  // Get a handle for our MVP and color uniforms
  mloc = glGetUniformLocation(shaderProgram, "M");
  mvploc = glGetUniformLocation(shaderProgram, "MVP");
  colorloc = glGetUniformLocation(shaderProgram, "color");
  camloc = glGetUniformLocation(shaderProgram, "cameraloc");
  ldirloc = glGetUniformLocation(shaderProgram, "lightdir");

  glm::vec4 lightcoeff(0.3, 0.7, 2.8, 50.5); // Lighting coeff, Ka, Kd, Ks, alpha
  GLuint lcoeloc = glGetUniformLocation(shaderProgram, "lightcoeff");
  glUniform4fv(lcoeloc, 1, &lightcoeff[0]);
}

void
RenderManager::SetView(glm::vec3 &camera, glm::vec3 &origin, glm::vec3 &up)
{ 
   glm::mat4 v = glm::lookAt(
                       camera, // Camera in world space
                       origin, // looks at the origin
                       up      // and the head is up
                 );
   view = v; 
   glUniform3fv(camloc, 1, &camera[0]);
   // Direction of light
   glm::vec3 lightdir = glm::normalize(camera - origin);   
   glUniform3fv(ldirloc, 1, &lightdir[0]);
};

void
RenderManager::SetUpWindowAndShaders()
{
  // start GL context and O/S window using the GLFW helper library
  if (!glfwInit()) {
    fprintf(stderr, "ERROR: could not start GLFW3\n");
    exit(EXIT_FAILURE);
  }

  glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 4);
  glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 0);
  glfwWindowHint(GLFW_OPENGL_FORWARD_COMPAT, GL_TRUE);
  glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_CORE_PROFILE);

  window = glfwCreateWindow(800, 800, "All City", NULL, NULL);
  if (!window) {
    fprintf(stderr, "ERROR: could not open window with GLFW3\n");
    glfwTerminate();
    exit(EXIT_FAILURE);
  }
  glfwMakeContextCurrent(window);
  // start GLEW extension handler
  //glfwExperimental = GL_TRUE;
  glfwInit();

  // get version info
  const GLubyte *renderer = glGetString(GL_RENDERER); // get renderer string
  const GLubyte *version = glGetString(GL_VERSION);   // version as a string
  printf("Renderer: %s\n", renderer);
  printf("OpenGL version supported %s\n", version);

  // tell GL to only draw onto a pixel if the shape is closer to the viewer
  glEnable(GL_DEPTH_TEST); // enable depth-testing
  glDepthFunc(GL_LESS); // depth-testing interprets a smaller value as "closer"

  const char* vertex_shader = GetVertexShader();
  const char* fragment_shader = GetFragmentShader();

  GLuint vs = glCreateShader(GL_VERTEX_SHADER);
  glShaderSource(vs, 1, &vertex_shader, NULL);
  glCompileShader(vs);
  int params = -1;
  glGetShaderiv(vs, GL_COMPILE_STATUS, &params);
  if (GL_TRUE != params) {
    fprintf(stderr, "ERROR: GL shader index %i did not compile\n", vs);
    _print_shader_info_log(vs);
    exit(EXIT_FAILURE);
  }

  GLuint fs = glCreateShader(GL_FRAGMENT_SHADER);
  glShaderSource(fs, 1, &fragment_shader, NULL);
  glCompileShader(fs);
  glGetShaderiv(fs, GL_COMPILE_STATUS, &params);
  if (GL_TRUE != params) {
    fprintf(stderr, "ERROR: GL shader index %i did not compile\n", fs);
    _print_shader_info_log(fs);
    exit(EXIT_FAILURE);
  }

  shaderProgram = glCreateProgram();
  glAttachShader(shaderProgram, fs);
  glAttachShader(shaderProgram, vs);
  glLinkProgram(shaderProgram);
  glUseProgram(shaderProgram);
}

void RenderManager::SetColor(double r, double g, double b)
{
   color[0] = r;
   color[1] = g;
   color[2] = b;
}

void RenderManager::MakeModelView(glm::mat4 &model)
{
   glm::mat4 modelview = projection * view * model;
   glm::mat4 M = model;
   glUniformMatrix4fv(mloc, 1, GL_FALSE, &M[0][0]);
   glUniformMatrix4fv(mvploc, 1, GL_FALSE, &modelview[0][0]);
}

void RenderManager::Render(ShapeType st, glm::mat4 model)
{
   int numPrimitives = 0;
   if (st == SPHERE)
   {
      glBindVertexArray(sphereVAO);
      numPrimitives = sphereNumPrimitives;
   }
   else if (st == CYLINDER)
   {
      glBindVertexArray(cylinderVAO);
      numPrimitives = cylinderNumPrimitives;
   }
   else if (st == CUBE)
   {
      glBindVertexArray(cubeVAO);
      numPrimitives = cubeNumPrimitives; 
   }
   MakeModelView(model);
   glUniform3fv(colorloc, 1, &color[0]);
   glDrawElements(GL_TRIANGLES, numPrimitives, GL_UNSIGNED_INT, NULL);
}

void SetUpVBOs(std::vector<float> &coords, std::vector<float> &normals,
               GLuint &points_vbo, GLuint &normals_vbo, GLuint &index_vbo)
{
  int numIndices = coords.size()/3;
  std::vector<GLuint> indices(numIndices);
  for(int i = 0; i < numIndices; i++)
    indices[i] = i;

  points_vbo = 0;
  glGenBuffers(1, &points_vbo);
  glBindBuffer(GL_ARRAY_BUFFER, points_vbo);
  glBufferData(GL_ARRAY_BUFFER, coords.size() * sizeof(float), coords.data(), GL_STATIC_DRAW);

  normals_vbo = 0;
  glGenBuffers(1, &normals_vbo);
  glBindBuffer(GL_ARRAY_BUFFER, normals_vbo);
  glBufferData(GL_ARRAY_BUFFER, normals.size() * sizeof(float), normals.data(), GL_STATIC_DRAW);

  index_vbo = 0;    // Index buffer object
  glGenBuffers(1, &index_vbo);
  glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, index_vbo);
  glBufferData(GL_ELEMENT_ARRAY_BUFFER, indices.size() * sizeof(GLuint), indices.data(), GL_STATIC_DRAW);
}

void RenderManager::SetUpGeometry()
{
  std::vector<float> sphereCoords;
  std::vector<float> sphereNormals;
  GetSphereData(sphereCoords, sphereNormals);
  sphereNumPrimitives = sphereCoords.size() / 3;
  GLuint sphere_points_vbo, sphere_normals_vbo, sphere_indices_vbo;
  SetUpVBOs(sphereCoords, sphereNormals, 
            sphere_points_vbo, sphere_normals_vbo, sphere_indices_vbo);

  std::vector<float> cylCoords;
  std::vector<float> cylNormals;
  GetCylinderData(cylCoords, cylNormals);
  cylinderNumPrimitives = cylCoords.size() / 3;
  GLuint cyl_points_vbo, cyl_normals_vbo, cyl_indices_vbo;
  SetUpVBOs(cylCoords, cylNormals, 
            cyl_points_vbo, cyl_normals_vbo, cyl_indices_vbo);

  std::vector<float> cubeCoords;
  std::vector<float> cubeNormals;
  GetCubeData(cubeCoords, cubeNormals);
  cubeNumPrimitives = cubeCoords.size() / 3;
  GLuint cube_points_vbo, cube_normals_vbo, cube_indices_vbo;
  SetUpVBOs(cubeCoords, cubeNormals,
            cube_points_vbo, cube_normals_vbo, cube_indices_vbo);

  GLuint vao[3];
  glGenVertexArrays(3, vao);

  glBindVertexArray(vao[SPHERE]);
  sphereVAO = vao[SPHERE];
  glBindBuffer(GL_ARRAY_BUFFER, sphere_points_vbo);
  glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 0, NULL);
  glBindBuffer(GL_ARRAY_BUFFER, sphere_normals_vbo);
  glVertexAttribPointer(1, 3, GL_FLOAT, GL_FALSE, 0, NULL);
  glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, sphere_indices_vbo);
  glEnableVertexAttribArray(0);
  glEnableVertexAttribArray(1);

  glBindVertexArray(vao[CYLINDER]);
  cylinderVAO = vao[CYLINDER];
  glBindBuffer(GL_ARRAY_BUFFER, cyl_points_vbo);
  glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 0, NULL);
  glBindBuffer(GL_ARRAY_BUFFER, cyl_normals_vbo);
  glVertexAttribPointer(1, 3, GL_FLOAT, GL_FALSE, 0, NULL);
  glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, cyl_indices_vbo);
  glEnableVertexAttribArray(0);
  glEnableVertexAttribArray(1);

  glBindVertexArray(vao[CUBE]);
  cubeVAO = vao[CUBE];
  glBindBuffer(GL_ARRAY_BUFFER, cube_points_vbo);
  glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 0, NULL);
  glBindBuffer(GL_ARRAY_BUFFER, cube_normals_vbo);
  glVertexAttribPointer(1, 3, GL_FLOAT, GL_FALSE, 0, NULL);
  glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, cube_indices_vbo);
  glEnableVertexAttribArray(0);
  glEnableVertexAttribArray(1);
}


int main()
{
  RenderManager rm;
  GLFWwindow *window = rm.GetWindow();

  glfwSetKeyCallback(window, KeyCallback);
  glfwSetInputMode(window, GLFW_STICKY_KEYS, 1);

  glm::vec3 origin(0, 0, 0);
  glm::vec3 up(0, 1, 0);

  while (!glfwWindowShouldClose(window))
  {
    glm::vec3 camera(0, 2, 10);
    rm.SetView(camera, origin, up);

    glClearColor(0.3, 0.3, 0.3, 1.0);
    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

    SetUpLevel(rm);

    SetUpCharacter(rm);

    glfwPollEvents();

    glfwSwapBuffers(window);
  }

  glfwTerminate();
  return 0;
}

glm::mat4 
RotateMatrix(float degrees, float x, float y, float z)
{
   glm::mat4 identity(1.0f);
   glm::mat4 rotation = glm::rotate(identity, 
                                    glm::radians(degrees), 
                                    glm::vec3(x, y, z));
   return rotation;
}

glm::mat4 
ScaleMatrix(double x, double y, double z)
{
   glm::mat4 identity(1.0f);
   glm::vec3 scale(x, y, z);
   return glm::scale(identity, scale);
}

glm::mat4 
TranslateMatrix(double x, double y, double z)
{
   glm::mat4 identity(1.0f);
   glm::vec3 translate(x, y, z);
   return glm::translate(identity, translate);
}

void
SetUpLevel(RenderManager &rm)
{
  glm::mat4 identity(1.0f);
  glm::mat4 translate = TranslateMatrix(0, -3, 0);

  rm.SetColor(1,0,0);
  rm.Render(RenderManager::CUBE, identity*translate);
}

void
SetUpCharacter(RenderManager &rm)
{
  glm::mat4 identity(1.0f);
  glm::mat4 translate = TranslateMatrix(0, 0, 0);

  rm.SetColor(0, 1, 0);
  rm.Render(RenderManager::SPHERE, identity*translate);
}

const char *
GetVertexShader()
{
   static char vertexShader[4096];
   strcpy(vertexShader, 
          "#version 400\n"
          "layout (location = 0) in vec3 vertex_position;\n"
          "layout (location = 1) in vec3 vertex_normal;\n"
          "uniform mat4 M;\n"
          "vec3 normal = normalize(mat3(transpose(inverse(M))) * vertex_normal);\n"
          "uniform mat4 MVP;\n"
          "uniform vec3 cameraloc;  // Camera position \n"
          "uniform vec3 lightdir;   // Lighting direction \n"
          "uniform vec4 lightcoeff; // Lighting coeff, Ka, Kd, Ks, alpha\n"
          "vec3 reflection, viewDir;\n"
          "float LdotN, diffuse, RdotV, specular;\n"
          "out float shading_amount;\n"
          "void main() {\n"
          "  vec4 position = vec4(vertex_position, 1.0);\n"
          "  gl_Position = MVP*position;\n"
          "  LdotN = dot(lightdir, normal);\n"
          "  diffuse = lightcoeff[1] * max(0.0, LdotN);\n"
          "  reflection = 2 * LdotN * normal - lightdir;\n"
          "  viewDir = cameraloc-vertex_position;\n"
          "  RdotV = dot(normalize(reflection), normalize(viewDir));\n"
          "  specular = abs(lightcoeff[2] * pow(max(0.0, RdotV), lightcoeff[3]));\n"
          "  shading_amount = lightcoeff[0] + diffuse + specular;\n"
          "}\n"
         );
   return vertexShader;
}

const char *
GetFragmentShader()
{
   static char fragmentShader[4096];
   strcpy(fragmentShader, 
          "#version 400\n"
          "uniform vec3 color;\n" 
          "in float shading_amount;\n"
          "out vec4 frag_color;\n"
          "void main() {\n"
          "  frag_color = vec4(color*shading_amount, 1.0);\n"
          "}\n"
         );
   return fragmentShader;
}

void
KeyCallback(GLFWwindow *window, int key, int scancode, int action, int mods)
{
  if (key == GLFW_KEY_SPACE && (action == GLFW_PRESS || action == GLFW_REPEAT))
  {
    cerr << "SPACE" << endl;
  }

  if (key == GLFW_KEY_A && (action == GLFW_PRESS || action == GLFW_REPEAT))
  {
    cerr << "A" << endl;
  }

  if (key == GLFW_KEY_D && (action == GLFW_PRESS || action == GLFW_REPEAT))
  {
    cerr << "D" << endl;
  }

  if (key == GLFW_KEY_W && (action == GLFW_PRESS || action == GLFW_REPEAT))
  {
    cerr << "W" << endl;
  }

  if (key == GLFW_KEY_S && (action == GLFW_PRESS || action == GLFW_REPEAT))  
  {
    cerr << "S" << endl;
  }
}
