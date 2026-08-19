fn main() {
    println!("cargo:rerun-if-changed=csrc/pfor_shim.cpp");
    println!("cargo:rerun-if-changed=vendor/FastPFor");
    // FastPFor is x86-SIMD only; the Rust pfor module is #[cfg(target_arch="x86_64")], so on
    // other arches skip the C++ build entirely (cell decode falls back to pure-Python pyfastpfor).
    if std::env::var("CARGO_CFG_TARGET_ARCH").as_deref() != Ok("x86_64") {
        return;
    }

    let mut b = cc::Build::new();
    b.cpp(true)
        .std("c++11")
        .flag_if_supported("-msse4.1")
        .opt_level(3)
        .include("vendor/FastPFor/headers")
        .file("csrc/pfor_shim.cpp");
    for e in std::fs::read_dir("vendor/FastPFor/src").unwrap() {
        let p = e.unwrap().path();
        if p.extension().and_then(|s| s.to_str()) == Some("cpp") {
            b.file(&p);
        }
    }
    b.compile("pforshim");

    let mut c_files = Vec::new();
    for e in std::fs::read_dir("vendor/FastPFor/src").unwrap() {
        let p = e.unwrap().path();
        if p.extension().and_then(|s| s.to_str()) == Some("c") {
            c_files.push(p);
        }
    }
    if !c_files.is_empty() {
        // streamvbyte.c/varintdecode.c provide C symbols referenced by CODECFactory;
        // compile them as C so the extern declarations link without C++ name mangling.
        let mut c = cc::Build::new();
        c.flag_if_supported("-msse4.1").opt_level(3);
        for p in c_files {
            c.file(p);
        }
        c.compile("pforc");
    }
}
