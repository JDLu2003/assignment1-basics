ifeq ($(MAKEFLAGS), -p)
	PROFILE := sample record 
endif

ifneq (,$(findstring r,$(MAKEFLAGS)))
	RUST_MODE := --release
	CLEAN_FOR_RUST_RELEASE := so_clean
endif

ifdef TS
	TEST_SUFFIX := /test_$(TS).py$(if $(FUNC),::$(FUNC))
endif

help:
	@echo "Usage:"
	@echo "  make test [TS=<test_suite>] [FUNC=<function_name>]"
	@echo "  make clean"
	@echo "  -p: profile the tests using samply"
	@echo "  -r: build the Rust extension in release mode"

clean: so_clean rust_clean

so_clean:
	rm cs336_basics/_rust_ext.*.so

rust_clean:
	cargo clean

rust_build: $(CLEAN_FOR_RUST_RELEASE)
	maturin develop $(RUST_MODE)

test: rust_build
	$(PROFILE) uv run pytest tests$(TEST_SUFFIX) -s

.PHONY: clean rust_clean rust_build test