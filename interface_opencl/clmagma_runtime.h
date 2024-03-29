/*
 * @Description: 
 * @Author: Shengle Lin
 * @Date: 2022-09-19 10:15:47
 * @LastEditors: Shengle Lin
 * @LastEditTime: 2022-09-19 14:46:03
 */
#ifndef CLMAGMA_RUNTIME_H
#define CLMAGMA_RUNTIME_H

#include <string>
#include <map>
#include <vector>

#include "common_magma.h"  // includes OpenCL, etc.
#include "error.h"


// ------------------------------------------------------------
class clmagma_runtime
{
public:
    // ------------------------------
    const static int MAX_DEVICES = 8;
    // const static int MAX_DEVICES = 1;
    
    // ------------------------------
    clmagma_runtime():
        m_bExternalContext (false),
        m_num_devices  ( 0 ),
        m_context      ( NULL )
    {
        for( int dev=0; dev < MAX_DEVICES; ++dev ) {
            m_devices[dev] = NULL;
        }
    }

    // ------------------------------
    ~clmagma_runtime()
    {
        quit();
    }
    
    // ------------------------------
    // void init( bool require_double=true );
    void init( bool require_double = false );
    void init(std::vector<cl_device_id> devices, cl_context context, bool require_double = false );
    void quit();
    int  compile_kernel( const char* kernel );
    int  compile_file( const char* infile, const char* outfile );
    void save_programs( std::vector< cl_program >& programs, const char* filename );
    void load_programs( int nfiles, const char* const* infiles, std::vector< cl_program >& programs );
    void archive_files( int nfiles, const char* const* infiles, const char* outfile );
    void load_kernels( int nfiles, const char* const* infiles );
    void load_kernels( const std::vector< cl_program >& programs );
    
    // ------------------------------
    cl_kernel get_kernel( const char* name )
    {
        //printf( "kernel: %s\n", name );
        cl_kernel k = m_kernels[ name ];
        if ( k == NULL ) {
            int err = compile_kernel( name );
            k = m_kernels[ name ];
            if ( err != 0 || k == NULL ) {
                fprintf( stderr, "Error: kernel '%s' not found\n", name );
                return NULL;
            }
        }
        return k;
    }
    
    // ------------------------------
    cl_platform_id get_platform()     const { return m_platform;    }
    int            get_num_devices()  const { return m_num_devices; }
    cl_context     get_context()      const { return m_context;     }
    cl_device_id*  get_devices()            { return m_devices;     }
    
    // ==============================
private:
    bool             m_bExternalContext;
    std::string      m_path;
    cl_platform_id   m_platform;
    cl_uint          m_num_devices;
    cl_context       m_context;
    cl_device_id     m_devices[ MAX_DEVICES ];
    std::map< std::string, cl_kernel > m_kernels;
    std::map< std::string, std::string > m_kernel_files;
};


// ------------------------------------------------------------
// global runtime
extern clmagma_runtime g_runtime;

#endif        //  #ifndef CLMAGMA_RUNTIME_H
