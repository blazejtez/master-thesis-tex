class HamiltonianOperatorCuPy:

    ... #implementation details

    def pre(self, v):
        v_cube = v.reshape((self.x_len, self.y_len, self.z_len))

        tex_obj = self.TextureFactory.texture_from_ndarray(v_cube)

        sur_obj = self.SurfaceFactory.initial_surface()

        return tex_obj, sur_obj

    def post(self, sur_out):
        v_out = self.SurfaceFactory.get_data(sur_out)
        v = cp.reshape(v_out, (self.x_len * self.y_len * self.z_len, 1),)

        return v

    def matvec(self, v: cp.ndarray):
        tex_obj, sur_obj = self.pre(v)
        potential = self.potential.operate_cupy(tex_obj, sur_obj, self.x_linspace, self.y_linspace, self.z_linspace)
        v1 = self.post(potential)

        tex_obj, sur_obj = self.pre(v)
        laplacian = self.laplacian.matcube_cupy_27(tex_obj, sur_obj)
        v2 = self.post(laplacian)
        return -v1 - 0.5*v2

    def matmat(self, V: cp.ndarray) -> cp.ndarray:
        V_out = cp.empty_like(V)
        for i in range(cp.size(V, 1)):
            v = V[:, i]
            v_out = self.matvec(v)
            V_out[:, i] = v_out.reshape(-1)
        return V_out