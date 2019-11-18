use super::*;

use super::memory::{Id, Tex, TextureSpec};

use conrod_core as cc;
use conrod_core::render::{Primitives};
use conrod_core::color::Rgba;
use conrod_core::text::{GlyphCache};




#[derive( Copy, Clone, Debug)]
pub struct Command {
    pub clip_rect : cc::Rect,
    pub elem_count : u32,
    pub texture_id : Option<Id<Tex>>,
    pub vtx_offset: u32,
}
impl Command {
    pub fn new() -> Self {
        Self{
            clip_rect: cc::Rect { x: cc::Range { start: 0.0, end: 0.0 }, y: cc::Range { start: 0.0, end: 0.0 } },
            elem_count: 0,
            texture_id: None,
            vtx_offset: 0
        }
    }
}
#[derive(Debug, Copy, Clone)]
pub struct UiVertex {
    pub pos: [f32; 2],
    pub uv: [f32; 2],
    pub mode: u32,
    pub color: u32,
}
#[derive(Debug)]
pub enum Error {
    UnhandledWidgetType,
    UncachedGlyph(u32),
}

const COL_NO_TEX: u32 = 2;
const FULL_TEX : u32 = 1;
const ALPHA_TEX: u32 = 0;

pub struct UiProcessor<'a>{
    pub image_map: cc::image::Map<Id<Tex>>,
    glyph_cache: GlyphCache<'a>,
    tex_id: Id<Tex>,
    cache_data: Vec<u8>,
    pub tex_updated: bool,

    key_map: std::collections::HashMap<conrod_core::image::Id, Id<Tex>>,
}


impl<'a> UiProcessor<'a> {
    pub fn from_glyphs(glyphs: impl Iterator<Item = (FontId, cc::text::PositionedGlyph)>) -> Self {
        let mut glyph_cache = GlyphCache::builder().build();
        for (font_id, glyph) in glyphs {
            glyph_cache.queue_glyph(font_id, glyph);
        }
        Self::from_glyph_cache_with_filled_queue(glyph_cache)
    }
    pub fn from_glyph_cache_with_filled_queue<'b: 'a>(glyph_cache: GlyphCache<'b>) -> Self {
        let (width, height) = glyph_cache.dimensions();
        let cache_data = vec![0; (width * height) as usize];
        let mut proc = Self {
            tex_id: Id::new(0),
            image_map: cc::image::Map::new(),
            glyph_cache,
            key_map: std::collections::HashMap::new(),
            tex_updated: true,
            cache_data
        };
        proc.update_gyph_cache(std::iter::empty());
        proc
    }
    pub fn get_image_tex_id(&self, image_id: conrod_core::image::Id) -> Option<Id<Tex>> {
        self.key_map.get(& image_id).map(|id| id.clone())
    }
    pub fn register_image(& mut self, id: Id<Tex>) -> conrod_core::image::Id {
        self.image_map.insert(id)
    }
    pub fn get_texture_spec(&mut self) -> TextureSpec {
        let (width, height) = self.glyph_cache.dimensions();
        self.tex_updated = false;
        TextureSpec {
            format: Format::R8Unorm,
            width,
            height,
            buffer: &self.cache_data[..]
        }
    }
    pub fn update_gyph_cache(&mut self, glyphs: impl Iterator<Item = (FontId, cc::text::PositionedGlyph)>) {
        info!("updating glyph cache");
        let mut gl: Vec<u32> = Vec::new();
        let mut gc = 0;
        for (font_id, glyph) in glyphs {
            gl.push(glyph.id().0);
            self.glyph_cache.queue_glyph(font_id, glyph);
            gc+=1;

        }
        let (width, _height) = self.glyph_cache.dimensions();

         while let Err(_) = {
             let cache_data: &mut[u8] = self.cache_data.as_mut();
             self.glyph_cache.cache_queued(|region, data|{
                 let mut c = 0usize;
                 let data_width = (region.max.x - region.min.x) as usize;
                 for row in region.min.y  .. region.max.y  {
                     & mut cache_data[(row * width + region.min.x) as usize ..(row * width + region.max.x) as usize]
                         .copy_from_slice(&data[data_width * c .. data_width * (c + 1) ]);
                     c += 1;
                 }
             })
         } {
             let (width, height) = self.glyph_cache.dimensions();
             let (new_width, new_height) = if width > height { (width, height * 2)} else { (width * 2, height) };
             self.glyph_cache.to_builder().dimensions(new_width, new_height).rebuild(& mut self.glyph_cache);
             self.cache_data.resize((new_width * new_height) as usize, 0);
         }
        use std::convert::TryInto;
        info!("cached {} glyphs, {}", gc, gl.iter().map(|&c|
            TryInto::<char>::try_into(c).unwrap()).collect::<String>());

        self.tex_updated = true;
    }
    pub fn process_primitives(&mut self, primitives: &mut Primitives, dpi_factor: f32, screen_width: f32, screen_height: f32)
                              -> Result<(Vec<Command>, Vec<UiVertex>), Error> {

        let mut cmds = Vec::new();
        let mut vertices = Vec::new();
        let mut current  = Command::new();
        //let ih = 2.0 * dpi_factor / screen_height;
        //let iw = -2.0 * dpi_factor /screen_width;

        let fresh = |cmd: Command | {
            let mut new : Command = Command::new();
            new.vtx_offset = cmd.vtx_offset + cmd.elem_count;
            return new;
        };
        use conrod_core::render::*;
        while let Some(primitive) = primitives.next() {
            if primitive.scizzor != current.clip_rect {
                if current.elem_count > 0 {
                    cmds.push(current);
                    current = fresh(current)
                }
                current.clip_rect = primitive.scizzor;
            }
            match primitive.kind {
                PrimitiveKind::Rectangle { color } => {
                    let color = normalize_rgba(color.to_rgb());
                    let mode = COL_NO_TEX;

                    let (l, r, b, t) = primitive.rect.l_r_b_t();
                    /*
                    let uv = [0.0, 0.0];
                    let p0 = [l as f32, t as f32];
                    let p1 = [r as f32, t as f32];
                    let p2 = [r as f32, b as f32];
                    let p3 = [l as f32, b as f32];
                    let mut v = |pos| {
                        vertices.push(UiVertex{ pos , uv, color, mode });
                        current.elem_count += 1;
                    };
                    v(p0);
                    v(p1);
                    v(p3);
                    v(p3);
                    v(p1);
                    v(p2);*/
                    let (u0, u1, v0, v1) = (0.0, 1.0, 0.0, 1.0);
                    let p0 = ([l as f32, t as f32], [u0 as f32, v0 as f32]);
                    let p1 = ([r as f32, t as f32], [u1 as f32, v0 as f32]);
                    let p2 = ([r as f32, b as f32], [u1 as f32, v1 as f32]);
                    let p3 = ([l as f32, b as f32], [u0 as f32, v1 as f32]);
                    let mut v = |(pos, uv): ([f32; 2], [f32; 2])| {
                        vertices.push(UiVertex{ pos , uv, color, mode });
                        current.elem_count += 1;
                    };
                    v(p0);
                    v(p1);
                    v(p3);
                    v(p3);
                    v(p1);
                    v(p2);
                },
                PrimitiveKind::TrianglesSingleColor { color, triangles } => {
                    let color = normalize_rgba(color);
                    let mode = COL_NO_TEX;
                    let uv = [0.0, 0.0];
                    let mut v = |pos : [f64; 2] | {
                        vertices.push(UiVertex{ pos : [pos[0] as f32, pos[1] as f32], uv, color, mode });
                        current.elem_count += 1;
                    };
                    for tri in triangles {
                        v(tri[0]);
                        v(tri[1]);
                        v(tri[2]);
                    }
                },
                PrimitiveKind::TrianglesMultiColor { triangles } => {
                    let mode = COL_NO_TEX;
                    let uv = [0.0, 0.0];
                    let mut v = |(p, c) : ([f64; 2], Rgba)|{
                        vertices.push(UiVertex{
                            pos : [p[0] as f32, p[1] as f32], uv,
                            color: normalize_rgba(c), mode
                        });
                        current.elem_count += 1;
                    };
                    for tri in triangles {
                        v(tri[0]);
                        v(tri[1]);
                        v(tri[2]);
                    }
                },
                PrimitiveKind::Image { image_id, color, source_rect } => {
                    let (color, mode) = match color {
                        Some(col) => (normalize_rgba(col.to_rgb()), FULL_TEX),
                        None => (std::u32::MAX, FULL_TEX)
                    };
                    let id = self.get_image_tex_id(image_id);
                    if id != current.texture_id {
                        if current.elem_count > 0 {
                            cmds.push(current);
                            current = fresh(current)
                        }
                        current.texture_id = id;
                        current.clip_rect = primitive.scizzor;
                    }
                    let (l, r, b, t) = primitive.rect.l_r_b_t();
                    let (u0, u1, v0, v1) = source_rect.map(|r| r.l_r_b_t())
                        .unwrap_or((0.0, 1.0, 0.0, 1.0));
                    let p0 = ([l as f32, t as f32], [u0 as f32, v0 as f32]);
                    let p1 = ([r as f32, t as f32], [u1 as f32, v0 as f32]);
                    let p2 = ([r as f32, b as f32], [u1 as f32, v1 as f32]);
                    let p3 = ([l as f32, b as f32], [u0 as f32, v1 as f32]);
                    let mut v = |(pos, uv): ([f32; 2], [f32; 2])| {
                        vertices.push(UiVertex{ pos , uv, color, mode });
                        current.elem_count += 1;
                    };
                    v(p0);
                    v(p1);
                    v(p3);
                    v(p3);
                    v(p1);
                    v(p2);
                },
                PrimitiveKind::Text { color, text, font_id } => {
                   // println!("color: {:?}", color);
                    let color = normalize_rgba(color.to_rgb());
                   // println!("color: {:?}", color);
                    let mode = ALPHA_TEX;
                    let id = Some(Id::new(0));
                    if id != current.texture_id {
                        if current.elem_count > 0 {
                            cmds.push(current);
                            current = fresh(current)
                        }
                        current.texture_id = id;
                        current.clip_rect = primitive.scizzor;
                    }
                   // println!("tex_id {:?}", current.texture_id);
                    let mut v = |(pos, uv): ([f32; 2], [f32; 2])| {
                        vertices.push(UiVertex{ pos , uv, color, mode });
                        current.elem_count += 1;
                    };

                    for glyph in text.positioned_glyphs(1.0 /* dpi_factor */){
                        let result = self.glyph_cache
                            .rect_for(font_id.index(), glyph).map_err(|_| Error::UncachedGlyph(glyph.id().0))?;
                        if let Some((texture_rect, window_rect)) = result {
                            // current.clip_rect = window_rect;

                                let l = window_rect.min.x as f32  - 0.5*  screen_width;
                                let r = l + (window_rect.max.x - window_rect.min.x) as f32;
                                let b = - window_rect.min.y as f32 + 0.5 * screen_height;
                                let t = b - (window_rect.max.y - window_rect.min.y) as f32;


                            let (u0, u1, v1, v0) = (texture_rect.min.x, texture_rect.max.x,
                                                    texture_rect.min.y, texture_rect.max.y);
                            let p0 = ([l as f32, t as f32], [u0 as f32, v0 as f32]);
                            let p1 = ([r as f32, t as f32], [u1 as f32, v0 as f32]);
                            let p2 = ([r as f32, b as f32], [u1 as f32, v1 as f32]);
                            let p3 = ([l as f32, b as f32], [u0 as f32, v1 as f32]);

                            v(p0);
                            v(p1);
                            v(p3);
                            v(p3);
                            v(p1);
                            v(p2);
                        }
                    }
                },
                PrimitiveKind::Other(_) => {
                    // Err(Error::UnhandledWidgetType)?;
                },
            }
        }
        if current.elem_count > 0 {
            cmds.push(current)
        }
        Ok((cmds, vertices))
    }
}

fn normalize_rgba(color: conrod_core::color::Rgba) -> u32 {
    let conrod_core::color::Rgba(r, g, b, a) = color;
    debug_assert!(r <= 1.0 && g <= 1.0 && b <= 1.0);
    const MAX : f32 = 255.0;
    ((r * MAX) as u32) << 0 | ((g * MAX) as u32) << 8 | ((b * MAX) as u32) << 16 | ((a * MAX) as u32) << 24
}
fn gamma_normalize_srgb(color: conrod_core::color::Rgba) -> u32 {
    let conrod_core::color::Rgba(r, g, b, a) = color;
    const MAX : f32 = 255.0;
    const INV_GAMMA : f32 = 1.0 / 2.2;
    fn component(c: f32) -> u32 {
        (c.powf(INV_GAMMA) * MAX) as u32
    }
    component(r) << 0 | component(g) << 8 | component(b) << 16 | ((a * MAX) as u32) << 24
}

pub struct GlyphWalker<'a> {
    prims: Primitives<'a>,
    dpi_factor: f32,
    font_id: FontId,
    current: Vec<cc::text::PositionedGlyph>
}
impl<'a> GlyphWalker<'a> {
    pub fn new(prims: Primitives<'a>, dpi_factor: f32) -> Self {
        Self{prims, dpi_factor, font_id : 0, current: Vec::new()}
    }
}
impl<'a> Iterator for GlyphWalker<'a> {
    type Item = (FontId, cc::text::PositionedGlyph);

    fn next(&mut self) -> Option<Self::Item> {
        use cc::render::*;
        if !self.current.is_empty() {
            return Some((self.font_id, self.current.remove(self.current.len() -1)));
        }
        while let Some(prim) = self.prims.next(){
            if let PrimitiveKind::Text{text, font_id, ..} = prim.kind {
                self.current.extend_from_slice(text.positioned_glyphs(self.dpi_factor));
                self.font_id = font_id.index();
                return self.next();
            }
        }
        None
    }
}


