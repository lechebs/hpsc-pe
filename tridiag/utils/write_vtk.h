#ifndef WRITE_VTK_H
#define WRITE_VTK_H

#include <stddef.h>
#include <stdint.h>
#include <stdio.h>

/*
 * VTK XML (ImageData, .vti) writer for scalar fields on a uniform
 * point-centred grid, using an appended raw binary block.
 *
 * The buffer is laid out z-major, matching ROW_MAJ(x, y, z):
 *
 *     data[(z * ny + y) * nx + x]
 *
 * Origin is (0, 0, 0) and the spacing is `h` in every direction.
 *
 * write_vtk       writes the whole nx * ny * nz volume.
 * write_vtk_slice writes the z = slice plane as an nx * ny * 1 image:
 *                 its physical position is z = slice * h.
 *
 * `name` is the name of the scalar array inside the VTK file.
 *
 * The appended block is uncompressed, native-endian binary data; the
 * byte_order attribute is set to match the host so that readers such as
 * ParaView interpret it correctly.
 */

static inline const char *vtk_byte_order(void)
{
    const uint16_t probe = 1;
    return *(const unsigned char *)&probe == 1 ? "LittleEndian" : "BigEndian";
}

static inline void write_vtk_image(const char *filename, const char *name,
                                   const double *data, int nx, int ny,
                                   int k0, int k1, double h)
{
    if (!data || nx <= 0 || ny <= 0 || k1 < k0) {
        fprintf(stderr, "write_vtk: invalid arguments\n");
        return;
    }

    const size_t nvalues =
        (size_t)nx * (size_t)ny * (size_t)(k1 - k0 + 1);
    const size_t nbytes = nvalues * sizeof(double);

    if (nbytes > (size_t)UINT32_MAX) {
        fprintf(stderr, "write_vtk: block too large for a UInt32 header\n");
        return;
    }

    FILE *fp = fopen(filename, "wb");
    if (!fp) {
        fprintf(stderr, "write_vtk: cannot open '%s'\n", filename);
        return;
    }

    fprintf(fp, "<?xml version=\"1.0\"?>\n");
    fprintf(fp, "<VTKFile type=\"ImageData\" version=\"1.0\" "
                "byte_order=\"%s\" header_type=\"UInt32\">\n",
            vtk_byte_order());
    fprintf(fp, "  <ImageData WholeExtent=\"0 %d 0 %d %d %d\" "
                "Origin=\"0 0 0\" Spacing=\"%.17g %.17g %.17g\">\n",
            nx - 1, ny - 1, k0, k1, h, h, h);
    fprintf(fp, "    <Piece Extent=\"0 %d 0 %d %d %d\">\n",
            nx - 1, ny - 1, k0, k1);
    fprintf(fp, "      <PointData Scalars=\"%s\">\n",
            name ? name : "scalar");
    fprintf(fp, "        <DataArray type=\"Float64\" Name=\"%s\" "
                "NumberOfComponents=\"1\" format=\"appended\" offset=\"0\"/>\n",
            name ? name : "scalar");
    fprintf(fp, "      </PointData>\n");
    fprintf(fp, "    </Piece>\n");
    fprintf(fp, "  </ImageData>\n");
    fprintf(fp, "  <AppendedData encoding=\"raw\">\n_");

    const uint32_t block_size = (uint32_t)nbytes;
    fwrite(&block_size, sizeof(block_size), 1, fp);
    fwrite(data, sizeof(double), nvalues, fp);

    fprintf(fp, "\n  </AppendedData>\n</VTKFile>\n");
    fclose(fp);
}

static inline void write_vtk(const char *filename, const char *name,
                             const double *data, int nx, int ny, int nz,
                             double h)
{
    if (nz <= 0) {
        fprintf(stderr, "write_vtk: invalid arguments\n");
        return;
    }
    write_vtk_image(filename, name, data, nx, ny, 0, nz - 1, h);
}

static inline void write_vtk_slice(const char *filename, const char *name,
                                   const double *data, int nx, int ny, int nz,
                                   int slice, double h)
{
    if (nz <= 0) {
        fprintf(stderr, "write_vtk_slice: invalid arguments\n");
        return;
    }

    if (slice < 0)
        slice = 0;
    if (slice >= nz)
        slice = nz - 1;

    const double *plane =
        data + (size_t)slice * (size_t)ny * (size_t)nx;

    write_vtk_image(filename, name, plane, nx, ny, slice, slice, h);
}

#endif /* WRITE_VTK_H */
